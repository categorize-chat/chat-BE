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
exports.renderMain = async (req, res) => {
  try {
    const user = await User.findById(req.user.id).populate('subscriptions');
    
    if (!user) {
      return res.status(404).json({
        isSuccess: false,
        code: 404,
        message: "사용자를 찾을 수 없습니다."
      });
    }
    
    // 구독 중인 모든 채팅방 정보 가져오기
    const rooms = await Room.find({
      _id: { $in: user.subscriptions }
    }).populate('owner participants')
      .populate({
        path: 'lastMessage',
        select: 'content createdAt',
        populate: {
          path: 'user',
          select: 'nickname profileUrl'
        }
      });

    // 각 채팅방별 총 메시지 수와 안 읽은 메시지 수 계산
    const channelsWithCounts = rooms.map(room => {
      // 해당 방의 읽은 메시지 수 찾기
      const roomId = room._id.toString();
      const readCount = user.readCounts && user.readCounts.get ? 
        user.readCounts.get(roomId) || 0 : 
        (user.readCounts && user.readCounts[roomId]) || 0;
      
      const unreadCount = Math.max(0, room.totalMessageCount - readCount);
      
      return {
        _id: room._id,
        channelName: room.channelName,
        description: room.description,
        channelId: room.channelId,
        owner: room.owner,
        participants: room.participants,
        createdAt: room.createdAt,
        totalMessageCount: room.totalMessageCount,
        unreadCount: unreadCount,
        lastMessage: room.lastMessage
      };
    });
    
    return res.json({
      isSuccess: true,
      code: 200,
      message: "채팅방 목록 조회 성공",
      result: { channels: channelsWithCounts }
    });
    
  } catch (error) {
    console.error('Render main error:', error);
    return res.status(500).json({
      isSuccess: false,
      code: 500,
      message: "서버 오류"
    });
  }
};

exports.getRooms = async (req, res, next) => {
  try {
    const channels = await Room.find()
      .limit(15)
      .populate('owner', 'nickname')
      .populate('participants', 'nickname profileUrl')
      .populate({
        path: 'lastMessage',
        select: 'content createdAt',
        populate: {
          path: 'user',
          select: 'nickname profileUrl'
        }
      })
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
    // 검색어 추출 및 검증
    const searchTerm = req.body.search;
    
    // 입력값 검증
    if (typeof searchTerm !== 'string') {
      return res.status(400).json({
        isSuccess: false,
        code: 400,
        message: "검색어는 문자열이어야 합니다."
      });
    }

    // 검색어 길이 제한
    if (searchTerm.length > 100) {
      return res.status(400).json({
        isSuccess: false,
        code: 400,
        message: "검색어가 너무 깁니다."
      });
    }

    // 특수문자 이스케이프 처리
    const escapedSearchTerm = searchTerm.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

    // 안전한 쿼리 실행
    const channels = await Room.find({ 
      channelName: { 
        $regex: new RegExp(escapedSearchTerm, 'i') 
      } 
    })
      .limit(50)
      .populate('owner', 'nickname email profileUrl')
      .populate('participants', 'nickname email profileUrl')
      .populate({
        path: 'lastMessage',
        select: 'content createdAt',
        populate: {
          path: 'user',
          select: 'nickname profileUrl'
        }
      })
      .sort({ createdAt: -1 });
    
    res.json({
      isSuccess: true,
      code: 200,
      message: "요청에 성공했습니다.",
      result: { channels },
    });
  } catch (error) {
    console.error('Search rooms error:', error);
    next(error);
  }
};

function validateAndSanitizeRoom(channelName, description) {
  // 기본적인 유효성 검사
  if (!channelName || typeof channelName !== 'string') {
    return { isValid: false, message: "올바른 채팅방 이름을 입력해주세요." };
  }

  if (description && typeof description !== 'string') {
    return { isValid: false, message: "올바른 설명을 입력해주세요." };
  }

  // 문자열 trim
  const trimmedName = channelName.trim();
  const trimmedDesc = description ? description.trim() : '';
  
  // 길이 검사
  if (trimmedName.length < 2 || trimmedName.length > 30) {
    return { isValid: false, message: "채팅방 이름은 2-30자 사이여야 합니다." };
  }

  if (trimmedDesc.length > 200) {
    return { isValid: false, message: "채팅방 설명은 200자를 초과할 수 없습니다." };
  }

  // 허용된 문자만 포함되어 있는지 검사 (알파벳, 숫자, 한글, 일부 특수문자만 허용)
  const nameRegex = /^[a-zA-Z0-9가-힣\s_.-]+$/;
  if (!nameRegex.test(trimmedName)) {
    return { isValid: false, message: "채팅방 이름에 허용되지 않은 문자가 포함되어 있습니다." };
  }

  // HTML 태그 이스케이프 처리
  const sanitizedName = trimmedName
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#x27;')
    .replace(/\//g, '&#x2F;');

  const sanitizedDesc = trimmedDesc
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#x27;')
    .replace(/\//g, '&#x2F;');

  return { 
    isValid: true, 
    sanitizedName, 
    sanitizedDesc 
  };
}

// client로부터 받은 이름으로 DB에 새 채팅방 생성
exports.createRoom = async (req, res, next) => {
  try {
    // 입력값 검증 및 sanitization
    const validation = validateAndSanitizeRoom(req.body.channelName, req.body.description);
    if (!validation.isValid) {
      return res.status(400).json({
        isSuccess: false,
        code: 400,
        message: validation.message
      });
    }

    const exist = await Room.findOne({ 
      channelName: validation.sanitizedName 
    });

    if (!exist) {
      try {
        const newRoom = await Room.create({
          channelName: validation.sanitizedName,
          description: validation.sanitizedDesc,
          owner: req.user.id,
          participants: [req.user.id]
        });

        await User.findByIdAndUpdate(
          req.user.id,
          { $addToSet: { subscriptions: newRoom._id } }
        );
        
        const io = req.app.get("io");
        io.of("/room").emit("newRoom", newRoom);

        res.json({
          isSuccess: true,
          code: 200,
          message: "요청에 성공했습니다.",
          result: {
            channelId: newRoom.channelId,
            channelName: newRoom.channelName,
            description: newRoom.description,
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
      res.status(409).json({
        isSuccess: false,
        code: 409,
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
    const roomId = req.params.id;
    const userId = req.user.id;
    const cursor = req.query.cursor;
    const limit = parseInt(req.query.limit) || 20;

    // 사용자의 구독 정보 확인
    const user = await User.findById(userId).select('subscriptions');
    if (!user) {
      return res.status(404).json({
        isSuccess: false,
        code: 404,
        message: "사용자를 찾을 수 없습니다."
      });
    }

    // 사용자가 해당 방을 구독하고 있는지 확인
    const isSubscribed = user.subscriptions.includes(roomId);
    if (!isSubscribed) {
      return res.status(403).json({
        isSuccess: false,
        code: 403,
        message: "구독하지 않은 채팅방입니다. 먼저 채팅방을 구독해주세요."
      });
    }

    // 방 정보 조회
    const room = await Room.findById(roomId)
      .populate('owner', 'nickname')
      .populate('participants', 'nickname profileUrl')
      .populate({
        path: 'lastMessage',
        select: 'content createdAt',
        populate: {
          path: 'user',
          select: 'nickname profileUrl'
        }
      });
      
    if (!room) {
      return res.status(404).json({
        isSuccess: false,
        code: 404,
        message: "존재하지 않는 방입니다."
      });
    }

    // 기본 쿼리: 해당 방의 메시지만 조회
    const baseQuery = { room: room._id };
    
    // cursor 기반 쿼리 설정
    if (cursor) {
      // cursor의 메시지가 해당 방의 것인지 확인
      const cursorMessage = await Chat.findOne({
        _id: cursor,
        room: room._id
      });

      if (!cursorMessage) {
        return res.status(400).json({
          isSuccess: false,
          code: 400,
          message: "유효하지 않은 cursor입니다."
        });
      }

      // createdAt 조건 추가
      baseQuery.createdAt = { $lt: cursorMessage.createdAt };
    }

    // 메시지 총 개수 확인
    const totalCount = await Chat.countDocuments(baseQuery);

    // 메시지 조회
    const messages = await Chat.find(baseQuery)
      .sort({ createdAt: -1 })
      .limit(limit)
      .populate('user', 'nickname profileUrl email');

    // 다음 페이지 존재 여부 확인
    const hasNextPage = totalCount > messages.length;
    
    // 응답 데이터 구성
    const responseData = {
      isSuccess: true,
      code: 200,
      message: "요청에 성공했습니다.",
      result: { 
        room,
        messages: messages.reverse(),
        nextCursor: hasNextPage ? messages[0]._id : null,
      },
    };

    return res.json(responseData);
  } catch (error) {
    console.error('Enter room error:', error);
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
    const roomId = req.params.roomId;
    const userId = req.user.id;

    // ObjectId 유효성 검사
    if (!mongoose.isValidObjectId(roomId)) {
      return res.status(400).json({
        isSuccess: false,
        code: 400,
        message: "유효하지 않은 방 ID입니다."
      });
    }

    // 사용자의 구독 정보 확인
    const user = await User.findById(userId).select('subscriptions');
    if (!user) {
      return res.status(404).json({
        isSuccess: false,
        code: 404,
        message: "사용자를 찾을 수 없습니다."
      });
    }

    // 사용자가 해당 방을 구독하고 있는지 확인
    const isSubscribed = user.subscriptions.includes(roomId);
    if (!isSubscribed) {
      return res.status(403).json({
        isSuccess: false,
        code: 403,
        message: "구독하지 않은 채팅방입니다. 먼저 채팅방을 구독해주세요."
      });
    }

    // 해당 방이 존재하는지 확인
    const room = await Room.findById(roomId)
      .populate({
        path: 'lastMessage',
        select: 'content createdAt',
        populate: {
          path: 'user',
          select: 'nickname profileUrl'
        }
      });

    if (!room) {
      return res.status(404).json({
        isSuccess: false,
        code: 404,
        message: "존재하지 않는 방입니다."
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
      room: roomId,
      user: userId,
      content: validation.sanitizedContent,
      createdAt: new Date(),
      topic: -1,
    });

    // 생성된 채팅 메시지에 user 정보를 포함하여 응답
    const populatedChat = await Chat.findById(chat._id)
      .populate('user', 'nickname profileUrl email');

    const io = req.app.get("io");
    io.of("/chat").to(roomId).emit("chat", populatedChat);
    
    res.json({
      isSuccess: true,
      code: 200,
      message: "메시지를 전송했습니다.",
      result: populatedChat
    });
  } catch (error) {
    console.error(error);
    next(error);
  }
};

// model 서버에 주제 요약 요청하고 그 결과값을 받아옴
exports.classifyChat = async (req, res, next) => {
  try {
    const { channelId, howmany, startMessageId } = req.body;
    const userId = req.user.id;

    if (!channelId) {
      return res.status(400).json({
        isSuccess: false,
        code: 400,
        message: "channelId가 필요합니다.",
      });
    }

    // ObjectId 유효성 검사
    if (!mongoose.isValidObjectId(channelId)) {
      return res.status(400).json({
        isSuccess: false,
        code: 400,
        message: "유효하지 않은 채널 ID입니다."
      });
    }

    // 사용자의 구독 정보 확인
    const user = await User.findById(userId).select('subscriptions');
    if (!user) {
      return res.status(404).json({
        isSuccess: false,
        code: 404,
        message: "사용자를 찾을 수 없습니다."
      });
    }

    // 사용자가 해당 방을 구독하고 있는지 확인
    const isSubscribed = user.subscriptions.includes(channelId);
    if (!isSubscribed) {
      return res.status(403).json({
        isSuccess: false,
        code: 403,
        message: "구독하지 않은 채팅방입니다. 먼저 채팅방을 구독해주세요."
      });
    }

    // 방 존재 여부 확인
    const room = await Room.findById(channelId)
      .populate({
        path: 'lastMessage',
        select: 'content createdAt',
        populate: {
          path: 'user',
          select: 'nickname profileUrl'
        }
      });

    if (!room) {
      return res.status(404).json({
        isSuccess: false,
        code: 404,
        message: "존재하지 않는 방입니다."
      });
    }

    // howmany가 없을 경우 채팅 100개만 분류
    const result = await classifyTopics(channelId, howmany || 100, startMessageId); 

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

exports.unsubscribeRoom = async (req, res, next) => {
  try {
    const roomId = req.params.roomId;
    const userId = req.user.id;

    // room의 participants와 user의 subscriptions 동시 업데이트
    const [room, user] = await Promise.all([
      Room.findByIdAndUpdate(
        roomId,
        { $pull: { participants: userId } },
        { new: true }
      ),
      User.findByIdAndUpdate(
        userId,
        { $pull: { subscriptions: roomId } },
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
      message: "채팅방 구독 해제에 성공했습니다.",
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
  if (trimmedNickname.length === 0 || trimmedNickname.length > 15) {
    return { isValid: false, message: "닉네임은 1-15자 사이여야 합니다." };
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

// 안 읽은 메시지 수 조회
exports.getUnreadCount = async (req, res) => {
  try {
    const userId = req.user.id;
    const user = await User.findById(userId);
    
    if (!user) {
      return res.status(404).json({
        isSuccess: false,
        code: 404,
        message: "사용자를 찾을 수 없습니다."
      });
    }
    
    // 특정 방의 안 읽은 메시지 수
    if (req.query.roomId) {
      const queryRoomId = req.query.roomId;
      const room = await Room.findById(queryRoomId)
        .populate({
          path: 'lastMessage',
          select: 'content createdAt',
          populate: {
            path: 'user',
            select: 'nickname profileUrl'
          }
        });
      
      if (!room) {
        return res.status(404).json({
          isSuccess: false,
          code: 404,
          message: "채팅방을 찾을 수 없습니다."
        });
      }
      
      const roomIdStr = room._id.toString();
      const readCountValue = user.readCounts && user.readCounts.get ? 
        user.readCounts.get(roomIdStr) || 0 : 
        (user.readCounts && user.readCounts[roomIdStr]) || 0;
      
      const unreadCount = Math.max(0, room.totalMessageCount - readCountValue);
      
      return res.json({
        isSuccess: true,
        code: 200,
        message: "안 읽은 메시지 수 조회 성공",
        result: { unreadCount }
      });
    }
    
    // 모든 구독 채팅방의 안 읽은 메시지 수
    const rooms = await Room.find({
      _id: { $in: user.subscriptions }
    });
    
    const unreadCounts = {};
    
    for (const room of rooms) {
      const roomIdStr = room._id.toString();
      const readCountValue = user.readCounts && user.readCounts.get ? 
        user.readCounts.get(roomIdStr) || 0 : 
        (user.readCounts && user.readCounts[roomIdStr]) || 0;
      
      unreadCounts[roomIdStr] = Math.max(0, room.totalMessageCount - readCountValue);
    }
    
    return res.json({
      isSuccess: true,
      code: 200,
      message: "모든 채팅방의 안 읽은 메시지 수 조회 성공",
      result: { unreadCounts }
    });
    
  } catch (error) {
    console.error('Unread count error:', error);
    return res.status(500).json({
      isSuccess: false,
      code: 500,
      message: "서버 오류"
    });
  }
};
