const Room = require("../schemas/room");
const Chat = require("../schemas/chat");
const User = require("../schemas/user");
const classifyTopics = require('../services/chatClassifier');

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
    const channels = await Room.find()
      .populate('owner', 'nickname')  // owner의 정보 포함
      .populate('participants', 'nickname profileImage');  // 참여자 정보 포함
    
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
      const newRoom = await Room.create({
        channelName: req.body.channelName,
        owner: req.user.id,  // 로그인한 사용자를 owner로 지정
        participants: [req.user.id]  // 생성자를 첫 참여자로 추가
      });

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
    } else {
      res.json({
        isSuccess: false,
        code: 404,
        message: "중복된 채팅방 이름입니다.",
      });
    }
  } catch (error) {
    console.error(error);
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
      return res.redirect("/?error=존재하지 않는 방입니다.");
    }

    // 참여자 목록에 사용자 추가
    if (!room.participants.includes(req.user.id)) {
      room.participants.push(req.user.id);
      await room.save();
    }

    const messages = await Chat.find({ room: room._id })
      .sort("createdAt")
      .populate('user', 'nickname profileImage snsId provider');  // user 정보를 더 자세히 포함
    
    return res.json({
      isSuccess: true,
      code: 200,
      message: "요청에 성공했습니다.",
      result: { 
        room,  // 방 정보 (owner, participants 포함)
        messages 
      },
    });
  } catch (error) {
    console.error(error);
    return next(error);
  }
};

exports.sendChat = async (req, res, next) => {
  try {
    const chat = await Chat.create({
      room: req.params.roomId,
      user: req.user.id,
      nickname: req.user.nickname,
      content: req.body.content,
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