const SocketIO = require("socket.io");
const Room = require("./schemas/room");
const Chat = require("./schemas/chat");
const User = require("./schemas/user");
const { verifyToken } = require('./utils/jwt');

function validateAndSanitizeChat(content) {
  if (!content || typeof content !== 'string') {
    return { isValid: false, message: "올바른 채팅 내용을 입력해주세요." };
  }

  const trimmedContent = content.trim();
  
  if (trimmedContent.length === 0 || trimmedContent.length > 1000) {
    return { isValid: false, message: "채팅은 1-1000자 사이여야 합니다." };
  }

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
    
    // 현재 사용자가 보고 있는 채팅방 ID를 저장
    let currentViewingRoom = null;
    
    socket.on("join", async (data) => {
      socket.join(data);
      currentViewingRoom = data;
      
      try {
        const room = await Room.findById(data);
        if (room && !room.participants.includes(socket.user.id)) {
          room.participants.push(socket.user.id);
          await room.save();
    
          // 채팅방 입장 시 읽음 상태 업데이트
          await updateUserReadCount(socket.user.id, data);
    
          const user = await User.findById(socket.user.id)
            .select('nickname profileUrl');
            
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

    // 사용자가 채팅방 화면을 보고 있을 때 읽음 상태 업데이트
    socket.on("view", async (roomId) => {
      currentViewingRoom = roomId;
      try {
        await updateUserReadCount(socket.user.id, roomId);
      } catch (error) {
        console.error('Update read count error:', error);
      }
    });
    
    // 사용자가 채팅방 화면을 벗어날 때
    socket.on("leave", () => {
      currentViewingRoom = null;
    });

    socket.on("message", async (data) => {
      try {
        const userId = socket.user.id;
        const roomId = data.room;
    
        // 사용자 구독 정보 확인
        const user = await User.findById(userId).select('subscriptions');
        if (!user) {
          return socket.emit("error", { 
            message: "사용자를 찾을 수 없습니다." 
          });
        }
    
        // 구독 여부 확인
        const isSubscribed = user.subscriptions.includes(roomId);
        if (!isSubscribed) {
          return socket.emit("error", { 
            message: "구독하지 않은 채팅방입니다. 먼저 채팅방을 구독해주세요." 
          });
        }
    
        // 방 존재 여부 확인
        const room = await Room.findById(roomId);
        if (!room) {
          return socket.emit("error", { 
            message: "존재하지 않는 방입니다." 
          });
        }
    
        // 메시지 검증
        const validation = validateAndSanitizeChat(data.content);
        if (!validation.isValid) {
          return socket.emit("error", { 
            message: validation.message 
          });
        }
    
        // 검증된 내용으로 채팅 생성
        const chat = await Chat.create({
          room: roomId,
          user: userId,
          content: validation.sanitizedContent,
          createdAt: new Date(),
        });
        
        // 1. Room의 totalMessageCount 증가 (원자적 연산)
        await Room.updateOne(
          { _id: roomId },
          { $inc: { totalMessageCount: 1 } }
        );
        
        // 2. 메시지 전송자의 읽음 상태 업데이트
        await updateUserReadCount(userId, roomId);
        
        // populate로 user 정보를 포함하여 조회
        const populatedChat = await Chat.findById(chat._id)
          .populate('user', 'nickname profileUrl email');
        
        // 채팅 발송
        io.of("/chat").to(roomId).emit("chat", populatedChat);
      } catch (error) {
        console.error('Message error:', error);
        socket.emit("error", { 
          message: "메시지 저장 중 오류가 발생했습니다." 
        });
      }
    });

    socket.on("disconnect", async () => {
      console.log("chat 네임스페이스 접속 해제");
      
      try {
        // 현재 소켓이 참여중인 모든 방 목록 가져오기
        const rooms = Array.from(socket.rooms);
        
        // socket.rooms에는 자신의 socket.id도 포함되어 있으므로 제외
        const chatRooms = rooms.filter(room => room !== socket.id);
        
        // 각 방에서 퇴장 처리
        for (const roomId of chatRooms) {
          if (!roomId) continue;  // roomId가 빈 문자열이면 스킵

          const room = await Room.findById(roomId);
          if (room) {
            room.participants = room.participants.filter(
              id => id.toString() !== socket.user.id
            );
            await room.save();

            socket.to(roomId).emit("exit", {
              type: "system",
              message: `${socket.user.nickname}님이 퇴장하셨습니다.`,
            });
          }
        }
      } catch (error) {
        console.error('Room leave error:', error);
      }
    });
  });
};

// 사용자의 읽음 상태 업데이트 함수
async function updateUserReadCount(userId, roomId) {
  const room = await Room.findById(roomId);
  if (!room) return;
  
  // 이미 해당 방에 대한 읽음 기록이 있는 경우
  const userWithReadCount = await User.findOne({
    _id: userId,
    "readCounts.room": roomId
  });
  
  if (userWithReadCount) {
    await User.updateOne(
      { _id: userId, "readCounts.room": roomId },
      { $set: { "readCounts.$.count": room.totalMessageCount } }
    );
  } else {
    // 처음 해당 방의 메시지를 읽는 경우
    await User.updateOne(
      { _id: userId },
      { $push: { readCounts: { room: roomId, count: room.totalMessageCount } } }
    );
  }
}