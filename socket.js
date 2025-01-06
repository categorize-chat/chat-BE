const SocketIO = require("socket.io");
const Room = require("./schemas/room");
const Chat = require("./schemas/chat");
const User = require("./schemas/user");
const { verifyToken } = require('./utils/jwt');
const user = require("./schemas/user");

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
      
      try {
        const room = await Room.findById(data);
        if (room && !room.participants.includes(socket.user.id)) {
          room.participants.push(socket.user.id);
          await room.save();
    
          const user = await User.findById(socket.user.id)
            .select('nickname profileImage');
            
          socket.to(data).emit("join", {
            type: "system",
            message: `${socket.user.nickname}님이 입장하셨습니다.`,
            user: user,
          });
        }
      } catch (error) {
        console.error('Room join error:', error);
      }
    });

    socket.on("message", async (data) => {
      try {
        // 메시지 검증
        const validation = validateAndSanitizeChat(data.content);
        if (!validation.isValid) {
          return socket.emit("error", { 
            message: validation.message 
          });
        }
    
        // 현재 유저 정보 조회
        const currentUser = await User.findById(socket.user.id);
        
        // 검증된 내용으로 채팅 생성
        const chat = await Chat.create({
          room: data.roomId,
          user: socket.user.id,  // ObjectId만 저장
          content: validation.sanitizedContent,
          createdAt: new Date(),
        });
    
        // populate로 user 정보를 포함하여 조회
        const populatedChat = {...chat, user: currentUser}
        
        io.of("/chat").to(data.roomId).emit("chat", populatedChat);
      } catch (error) {
        console.error(error);
        socket.emit("error", { message: "메시지 저장 중 오류가 발생했습니다." });
      }
    });

    socket.on("disconnect", async () => {
      console.log("chat 네임스페이스 접속 해제");
      const { referer } = socket.request.headers;
      const roomId = new URL(referer).pathname.split("/").at(-1);
      
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