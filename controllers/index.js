const Room = require("../schemas/room");
const Chat = require("../schemas/chat");
const User = require("../schemas/user");
const classifyTopics = require('../services/chatClassifier');
const mongoose = require('mongoose');

// client로부터 받은 닉네임으로 DB에 새로운 User 생성
exports.registerUser = async (req, res, next) => { 
  try {
    const exist = await User.findOne({ nickname: req.body.nickname }); // 닉네임 중복 확인

    if (!exist) {
      const newUser = await User.create({
        nickname: req.body.nickname,
      });
      res.json({ // json 형식으로 전달
        isSuccess: true, // 성공 여부 (true/false)
        code: 200, // 응답 코드
        message: "요청에 성공했습니다.", // 응답 메세지
        result: {
          userId: newUser.userId,
          nickname: newUser.nickname,
        },
      });
    } else { // 중복된 닉네임일 경우 경고
      res.json({
        isSuccess: false,
        code: 404,
        message: "중복된 유저 이름입니다.",
      });
    }
  } catch (error) {
    console.error(error);
    next(error);
  }
};

// 생성된 모든 채팅방들의 목록을 전달
exports.renderMain = async (req, res, next) => {
  try {
    const user = await User.findById(req.user.id).populate('subscriptions');
    const channels = await Room.find({
      _id: { $in: user.subscriptions }
    })
      .populate('owner', 'nickname')
      .populate('participants', 'nickname profileImage');
    
    res.json({
      isSuccess: true,
      code: 200,
      message: "요청에 성공했습니다.",
      result: { channels },
    });
  } catch (error) {
    console.error(error);
    next(error);
  }
};

exports.getRooms = async (req, res, next) => {
  try {
    const channels = await Room.find()
      .populate('owner', 'nickname')
      .populate('participants', 'nickname profileImage')
      .sort({ createdAt: -1 });
    
    res.json({
      isSuccess: true,
      code: 200,
      message: "요청에 성공했습니다.",
      result: { channels },
    });
  } catch (error) {
    console.error(error);
    next(error);
  }
};

exports.searchRooms = async (req, res, next) => {
  try {
    const { search } = req.body.search;
    const channels = await Room.find({ channelName: { $regex: search, $options: 'i' } })
      .populate('owner', 'nickname')
      .populate('participants', 'nickname profileImage')
      .sort({ createdAt: -1 });
    
    res.json({
      isSuccess: true,
      code: 200,
      message: "요청에 성공했습니다.",
      result: { channels },
    });
  } catch (error) {
    console.error(error);
    next(error);
  }
};

// client로부터 받은 이름으로 DB에 새 채팅방 생성
exports.createRoom = async (req, res, next) => {
  try {
    const exist = await Room.findOne({ channelName: req.body.channelName }); // 채팅방 이름 중복 확인

    if (!exist) {
      try {
        // 1. 새로운 방 생성
        const newRoom = await Room.create({
          channelName: req.body.channelName,
          owner: req.user.id,
          participants: [req.user.id]
        });

        // 2. 유저의 구독 목록에 새로운 방 추가
        await User.findByIdAndUpdate(
          req.user.id,
          { $addToSet: { subscriptions: newRoom._id } }
        );
        
        const io = req.app.get("io");
        io.of("/room").emit("newRoom", newRoom); // 새로운 채팅방이 생성되었음을 모든 클라이언트에게 알림

        res.json({
          isSuccess: true,
          code: 200,
          message: "요청에 성공했습니다.",
          result: {
            channelId: newRoom.channelId,
            channelName: newRoom.channelName,
            owner: newRoom.owner,
            participants: newRoom.participants
          },
        });
      } catch (error) {
        console.error('Room Creation Error:', error);
        res.status(500).json({
          isSuccess: false,
          code: 500,
          message: "채팅방 생성 중 오류가 발생했습니다."
        });
      }
    } else {
      res.json({
        isSuccess: false,
        code: 404,
        message: "중복된 채팅방 이름입니다.",
      });
    }
  } catch (error) {
    console.error('Create Room Error:', error);
    next(error);
  }
};

// 해당 채팅방의 모든 채팅 전달
exports.enterRoom = async (req, res, next) => {
  try {
    const room = await Room.findOne({ _id: req.params.id })
      .populate('owner', 'nickname')
      .populate('participants', 'nickname profileImage');
      
    if (!room) {
      return res.status(404).json({
        isSuccess: false,
        code: 404,
        message: "존재하지 않는 방입니다."
      });
    }

    const messages = await Chat.find({ room: room._id })
      .sort("createdAt")
      .populate('user', 'nickname profileImage email');
    
    const isSubscribed = room.participants.some(p => p._id.toString() === req.user.id);
    
    return res.json({
      isSuccess: true,
      code: 200,
      message: "요청에 성공했습니다.",
      result: { 
        room,
        messages,
        isSubscribed
      },
    });
  } catch (error) {
    console.error(error);
    return next(error);
  }
};

function validateAndSanitizeChat(content) {
  // 기본적인 유효성 검사
  if (!content || typeof content !== 'string') {
    return { isValid: false, message: "올바른 채팅 내용을 입력해주세요." };
  }

  // 문자열 trim
  const trimmedContent = content.trim();
  
  // 길이 검사 (예: 최대 1000자)
  if (trimmedContent.length === 0 || trimmedContent.length > 1000) {
    return { isValid: false, message: "채팅은 1-1000자 사이여야 합니다." };
  }

  // HTML 태그 이스케이프 처리
  const sanitizedContent = trimmedContent
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#x27;')
    .replace(/\//g, '&#x2F;');

  return { isValid: true, sanitizedContent };
}

exports.sendChat = async (req, res, next) => {
  try {
    // ObjectId 유효성 검사
    if (!mongoose.isValidObjectId(req.params.roomId)) {
      return res.status(400).json({
        isSuccess: false,
        code: 400,
        message: "유효하지 않은 방 ID입니다."
      });
    }

    // 채팅 내용 검증 및 sanitization
    const validation = validateAndSanitizeChat(req.body.content);
    if (!validation.isValid) {
      return res.status(400).json({
        isSuccess: false,
        code: 400,
        message: validation.message
      });
    }

    const chat = await Chat.create({
      room: req.params.roomId,
      user: req.user.id,
      nickname: req.user.nickname,
      content: validation.sanitizedContent,
      createdAt: new Date(),
      topic: -1,
    });

    // 생성된 채팅 메시지에 user 정보를 포함하여 응답
    const populatedChat = await Chat.findById(chat._id)
      .populate('user', 'nickname profileImage snsId provider');

    const io = req.app.get("io");
    io.of("/chat").to(req.params.id).emit("chat", populatedChat);
    res.json(populatedChat);
  } catch (error) {
    console.error(error);
    next(error);
  }
};

// model 서버에 주제 요약 요청하고 그 결과값을 받아옴
exports.classifyChat = async (req, res, next) => {
  try {
    const { channelId, howmany } = req.body;
    if (!channelId) {
      return res.json({
        isSuccess: false,
        code: 400,
        message: "channelId가 필요합니다.",
      });
    }

    // howmany가 없을 경우 채팅 100개만 분류
    const result = await classifyTopics(channelId, howmany || 100); 

    const io = req.app.get("io");
    io.of("/chat").to(channelId).emit("summary", result);

    return res.json({
      isSuccess: true,
      code: 200,
      message: "요청에 성공했습니다.",
      result
    });

  } catch (error) {
    console.error(error);
    next(error);
  }
};

exports.subscribeRoom = async (req, res, next) => {
  try {
    const roomId = req.params.roomId;
    const userId = req.user.id;

    // room의 participants와 user의 subscriptions 동시 업데이트
    const [room, user] = await Promise.all([
      Room.findByIdAndUpdate(
        roomId,
        { $addToSet: { participants: userId } },
        { new: true }
      ),
      User.findByIdAndUpdate(
        userId,
        { $addToSet: { subscriptions: roomId } },
        { new: true }
      )
    ]);

    if (!room || !user) {
      return res.status(404).json({
        isSuccess: false,
        code: 404,
        message: "채팅방 또는 사용자를 찾을 수 없습니다."
      });
    }

    res.json({
      isSuccess: true,
      code: 200,
      message: "채팅방 구독에 성공했습니다.",
      result: { room, user }
    });

  } catch (error) {
    console.error(error);
    next(error);
  }
};

// 닉네임 유효성 검사를 위한 함수
function validateAndSanitizeNickname(nickname) {
  // 기본적인 유효성 검사
  if (!nickname || typeof nickname !== 'string') {
    return { isValid: false, message: "올바른 닉네임을 입력해주세요." };
  }

  // 문자열 trim
  const trimmedNickname = nickname.trim();
  
  // 길이 검사
  if (trimmedNickname.length === 0 || trimmedNickname.length > 30) {
    return { isValid: false, message: "닉네임은 1-30자 사이여야 합니다." };
  }

  // 허용된 문자만 포함되어 있는지 검사 (알파벳, 숫자, 한글, 일부 특수문자만 허용)
  const nicknameRegex = /^[a-zA-Z0-9가-힣_.-]+$/;
  if (!nicknameRegex.test(trimmedNickname)) {
    return { isValid: false, message: "닉네임에 허용되지 않은 문자가 포함되어 있습니다." };
  }

  return { isValid: true, sanitizedNickname: trimmedNickname };
}

// 유저 설정 조회
exports.getUserSettings = async (req, res, next) => {
  try {
    // ObjectId 검증
    if (!mongoose.isValidObjectId(req.user.id)) {
      return res.status(400).json({
        isSuccess: false,
        code: 400,
        message: "유효하지 않은 사용자 ID입니다."
      });
    }

    const user = await User.findById(req.user.id);
    if (!user) {
      return res.status(404).json({
        isSuccess: false,
        code: 404,
        message: "사용자를 찾을 수 없습니다."
      });
    }

    res.json({
      isSuccess: true,
      code: 200,
      message: "요청에 성공했습니다.",
      result: user
    });
  } catch (error) {
    console.error(error);
    next(error);
  }
};

// 닉네임 변경
exports.updateUserNickname = async (req, res, next) => {
  try {
    const { nickname } = req.body;

    // ObjectId 검증
    if (!mongoose.isValidObjectId(req.user.id)) {
      return res.status(400).json({
        isSuccess: false,
        code: 400,
        message: "유효하지 않은 사용자 ID입니다."
      });
    }
    
    // 닉네임 유효성 검사 및 sanitization
    const validation = validateAndSanitizeNickname(nickname);
    if (!validation.isValid) {
      return res.status(400).json({
        isSuccess: false,
        code: 400,
        message: validation.message
      });
    }

    // 닉네임 업데이트 - sanitized된 닉네임 사용
    const updatedUser = await User.findByIdAndUpdate(
      req.user.id,
      { nickname: validation.sanitizedNickname },
      { 
        new: true,
        runValidators: true // 스키마 레벨의 유효성 검사 실행
      }
    );

    if (!updatedUser) {
      return res.status(404).json({
        isSuccess: false,
        code: 404,
        message: "사용자를 찾을 수 없습니다."
      });
    }

    res.json({
      isSuccess: true,
      code: 200,
      message: "닉네임이 성공적으로 변경되었습니다.",
      result: {
        nickname: updatedUser.nickname
      }
    });
  } catch (error) {
    // MongoDB 에러 처리
    if (error.name === 'ValidationError') {
      return res.status(400).json({
        isSuccess: false,
        code: 400,
        message: "닉네임 형식이 올바르지 않습니다."
      });
    }

    console.error(error);
    next(error);
  }
};