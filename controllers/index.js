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
    const channels = await Room.find(); // DB에서 지금까지 생성된 채팅방 조회
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
    const room = await Room.findOne({ channelId: req.params.id }); // 유효한 채널ID인지 검증
    if (!room) {
      return res.redirect("/?error=존재하지 않는 방입니다.");
    }

    const messages = await Chat.find({ room: room.channelId }).sort("createdAt");
    return res.json({
      isSuccess: true,
      code: 200,
      message: "요청에 성공했습니다.",
      result: { messages },
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
      nickname: req.body.nickname,
      content: req.body.content,
      createdAt: new Date(),
      topic: -1,
    });
    const io = req.app.get("io");
    io.of("/chat").to(req.params.id).emit("chat", chat);
    res.json(chat);
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
