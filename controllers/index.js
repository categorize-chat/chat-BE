const Room = require("../schemas/room");
const Chat = require("../schemas/chat");
const User = require("../schemas/user");
const classifyTopics = require('../services/chatClassifier');

exports.registerUser = async (req, res, next) => {
  console.log(`body: ${req.body}`)
  try {
    const exist = 0; //await User.findOne({ nickname: req.body.nickname }); for debug

    if (!exist) {
      const newUser = await User.create({
        nickname: req.body.nickname,
      });

      res.json({
        isSuccess: true, // 성공 여부 (Strue/false)
        code: 200, // 응답 코드
        message: "요청에 성공했습니다.", // 응답 메세지
        result: {
          userId: newUser.userId,
          nickname: newUser.nickname,
        },
      });
    } else {
      res.json({
        isSuccess: false, // 성공 여부 (Strue/false)
        code: 404, // 응답 코드
        message: "중복된 유저이름입니다.", // 응답 메세지
      });
    }
  } catch (error) {
    console.error(error);
    next(error);
  }
};

exports.renderMain = async (req, res, next) => {
  try {
    const channels = await Room.find();
    console.log(channels); // debug
    res.json({
      isSuccess: true, // 성공 여부 (true/false)
      code: 200, // 응답 코드
      message: "요청에 성공했습니다.", // 응답 메세지
      result: { channels },
    });
  } catch (error) {
    console.error(error);
    next(error);
  }
};

exports.createRoom = async (req, res, next) => {
  try {
    const exist = await Room.findOne({ channelName: req.body.channelName });

    if (!exist) {
      const newRoom = await Room.create({
        channelName: req.body.channelName,
      });

      const io = req.app.get("io");
      io.of("/room").emit("newRoom", newRoom);

      res.json({
        isSuccess: true, // 성공 여부 (Strue/false)
        code: 200, // 응답 코드
        message: "요청에 성공했습니다.", // 응답 메세지
        result: {
          channelId: newRoom.channelId,
          channelName: newRoom.channelName,
        },
      });
    } else {
      res.json({
        isSuccess: false, // 성공 여부 (Strue/false)
        code: 404, // 응답 코드
        message: "중복된 채널명입니다.", // 응답 메세지
      });
    }
  } catch (error) {
    console.error(error);
    next(error);
  }
};

exports.enterRoom = async (req, res, next) => {
  try {
    const room = await Room.findOne({ _id: req.params.id });
    if (!room) {
      return res.redirect("/?error=존재하지 않는 방입니다.");
    }
    const io = req.app.get("io");
    const { rooms } = io.of("/chat").adapter;
    console.log(rooms, rooms.get(req.params.id), rooms.get(req.params.id));

    const messages = await Chat.find({ room: room._id }).sort("createdAt");
    console.log(messages);
    console.log("debug");
    return res.json({
      isSuccess: true, // 성공 여부 (true/false)
      code: 200, // 응답 코드
      message: "요청에 성공했습니다.", // 응답 메세지
      result: { messages },
    });
  } catch (error) {
    console.error(error);
    return next(error);
  }
};

exports.sendChat = async (req, res, next) => {
  try {
    console.log(req.params.id);
    const chat = await Chat.create({
      room: req.params.roomId,
      nickname: req.body.nickname,
      content: req.body.content,
      createdAt: new Date(),
    });
    req.app.get("io").of("/chat").to(req.params.id).emit("chat", chat);
    console.log(chat);
    res.json(chat);
  } catch (error) {
    console.error(error);
    next(error);
  }
};

exports.classifyChat = async (req, res, next) => {
  try {
    const { channelId } = req.body;
    if (!channelId) {
      return res.status(400).json({ error: 'channelId is required' });
    }
    // const result = await classifyTopics(channelId);
    const result = {
      messages: [
        {
          nickname: "user1",
          createdAt: "2024-05-21T14:52:52.614581",
          content: "안녕하세요",
          topic: 1
        },
        {
          nickname: "user1",
          createdAt: "2024-05-21T14:52:52.614581",
          content: "안녕하세요",
          topic: 2,
        },
        {
          nickname: "user1",
          createdAt: "2024-05-21T14:52:52.614581",
          content: "안녕하세요",
          topic: 0,
        }
      ],
      summary: {
        0: {
          keywords: ['슈마메', '파이데이'],
          content: "....."
        }
      }
    }
    res.status(200).json({ 
      isSuccess: true, // 성공 여부 (Strue/false)
      code: 200, // 응답 코드
      message: "요청에 성공했습니다.", // 응답 메세지
      result 
    });
  } catch (error) {
    console.error('Error in chat classification:', error);
    res.status(500).json({ error: 'An error occurred during chat classification' });
  }
};
