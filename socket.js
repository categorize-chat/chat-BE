const SocketIO = require("socket.io");
const Room = require("./schemas/room");
const Chat = require("./schemas/chat");
const User = require("./schemas/user");
const { verifyToken } = require('./utils/jwt');

module.exports = (server, app) => {
  const io = SocketIO(server, {
    cors: {
      origin: process.env.CLIENT_URL,
      credentials: true
    },
    path: "/socket.io"
  });

  app.set("io", io);
  const room = io.of("/room");
  const chat = io.of("/chat");

  // Socket 인증 미들웨어
  const socketAuthMiddleware = (socket, next) => {
    const token = socket.handshake.auth.token;
    if (!token) {
      return next(new Error('Authentication error'));
    }

    const { valid, decoded } = verifyToken(token);
    if (!valid) {
      return next(new Error('Invalid token'));
    }

    socket.user = decoded;
    next();
  };

  chat.use(socketAuthMiddleware);

  room.on("connection", (socket) => {
    console.log("room 네임스페이스에 접속");
    socket.on("disconnect", () => {
      console.log("room 네임스페이스 접속 해제");
    });
  });

  chat.on("connection", (socket) => {
    console.log("chat 네임스페이스에 접속");
    
    socket.on("join", async (data) => {
      socket.join(data);
      
      // 방 참여자 목록에 추가
      try {
        const room = await Room.findById(data);
        if (room && !room.participants.includes(socket.user.id)) {
          room.participants.push(socket.user.id);
          await room.save();
    
          // 참여자 정보를 포함하여 이벤트 전송
          const user = await User.findById(socket.user.id)
            .select('nickname profileImage');
            
          socket.to(data).emit("join", {
            type: "system",
            message: `${socket.user.nickname}님이 입장하셨습니다.`,
            user: user,  // 참여한 유저 정보 포함
          });
        }
      } catch (error) {
        console.error('Room join error:', error);
      }
    });

    socket.on("message", async (data) => {
      try {
        const chat = await Chat.create({
          room: data.roomId,
          user: socket.user.id,
          nickname: socket.user.nickname,
          content: data.content,
          createdAt: new Date(),
        });
        io.of("/chat").to(data.roomId).emit("chat", chat);
      } catch (error) {
        console.error(error);
        socket.emit("error", { message: "메시지 저장 중 오류가 발생했습니다." });
      }
    });

    socket.on("disconnect", async () => {
      console.log("chat 네임스페이스 접속 해제");
      const { referer } = socket.request.headers;
      const roomId = new URL(referer).pathname.split("/").at(-1);
      
      // 방 참여자 목록에서 제거
      try {
        const room = await Room.findById(roomId);
        if (room) {
          room.participants = room.participants.filter(id => id.toString() !== socket.user.id);
          await room.save();
        }
      } catch (error) {
        console.error('Room leave error:', error);
      }

      socket.to(roomId).emit("exit", {
        user: "system",
        chat: `${socket.user.nickname}님이 퇴장하셨습니다.`,
      });
    });
  });
};