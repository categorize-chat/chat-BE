const Room = require("../schemas/room");
const Chat = require("../schemas/chat");
const User = require("../schemas/user");
const classifyTopics = require('../services/chatClassifier');

exports.registerUser = async (req, res, next) => { // 유저 등록
  try {
    const exist = await User.findOne({ nickname: req.body.nickname }); // 닉네임 중복 확인

    if (!exist) { // 새로운 닉네임일 경우 계정 생성
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

exports.renderMain = async (req, res, next) => { // 채팅방 조회
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

exports.createRoom = async (req, res, next) => { // 채팅방 생성
  try {
    const exist = await Room.findOne({ channelName: req.body.channelName }); // 채팅방 이름 중복 확인

    if (!exist) { // 중복 아닐 경우 채팅방 생성
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
    } else { // 중복일 경우 경고
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

exports.enterRoom = async (req, res, next) => { // 채팅 내역 조회
  try {
    const room = await Room.findOne({ _id: req.params.id }); // 유효한 채널ID인지 검증
    if (!room) {
      return res.redirect("/?error=존재하지 않는 방입니다.");
    }

    const messages = await Chat.find({ room: room._id }).sort("createdAt"); // room의 채팅들을 생성순으로 정렬
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

exports.sendChat = async (req, res, next) => { // 채팅 보내기
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

exports.classifyChat = async (req, res, next) => { // 주제 요약하기 요청
  try {
    const { channelId, howmany } = req.body;
    if (!channelId) {
      return res.json({
        isSuccess: false,
        code: 400,
        message: "channelId가 필요합니다.",
      });
    }

    // howmany가 없을 경우 기본값 100으로 설정
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
