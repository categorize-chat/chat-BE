const SocketIO = require("socket.io");
const Room = require("./schemas/room");
const Chat = require("./schemas/chat");
const User = require("./schemas/user");
const jwt = require('jsonwebtoken');
const sanitizeHtml = require('sanitize-html');

function validateChat(content) {
  if (!content || typeof content !== 'string') {
    return {
      isValid: false, 
      message: "올바른 채팅 형식이 아닙니다." 
    };
  }

  const chat = content.trim();
  
  if (chat.length === 0) {
    return { 
      isValid: false, 
      message: "채팅 내용을 입력해주세요." 
    };
  }
  
  if (chat.length > 1000) {
    return { 
      isValid: false, 
      message: "채팅은 1000자 이하여야 합니다." 
    };
  }

  // XSS 공격 방지
  // 기존의 replace 방식보다 전문가들이 제공하는 모듈이 훨씬 더 안전하다고 생각해서 변경함
  const safeChat = sanitizeHtml(chat);

  return { isValid: true, safeChat };
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

  const socketAuthMiddleware = (socket, next) => {
    const token = socket.handshake.auth.token;
    if (!token) {
      return next(new Error('토큰 없음'));
    }

    try {
      const decoded = jwt.verify(token, process.env.JWT_SECRET);
      socket.user = decoded;
    } catch (error) {
      return next(new Error('유효하지 않은 토큰'));
    }
    next();
  };

  chat.use(socketAuthMiddleware);

  room.on("connection", (socket) => {
    socket.on("disconnect", () => {
    });
  });

  chat.on("connection", (socket) => {
    
    socket.on("error", (error) => {
      console.error("소켓 에러:", error);
    });
    
    let currentViewingRoom = null;
    
    socket.on("join", async (data) => {
      data.forEach(roomId => {  
        console.log("채팅방 입장:", roomId, socket.user?.nickname);
        socket.join(roomId);
      });
    });

    // 사용자가 해당 채팅방 화면을 보고 있을 때 읽음 상태 업데이트 해야함
    socket.on("view", async (roomId) => {
        currentViewingRoom = roomId;
      try {
        await updateReadCount(socket.user.id, roomId);
      } catch (error) {
        console.error('채팅방 읽음 상태 업데이트 오류:', error);
      }
    });
    
    socket.on("leave", () => {
      currentViewingRoom = null;
    });

    socket.on("message", async (data) => {
      
      try {
        if (!data || !data.room || !data.content) {
          console.error('잘못된 메시지 형식:', data);
          return socket.emit("error", { 
            message: "메시지 형식이 올바르지 않습니다." 
          });
        }
        
        const userId = socket.user.id;
        const roomId = data.room;
    
        const user = await User.findById(userId).select('subscriptions');
        if (!user) {
          console.error('사용자를 찾을 수 없음:', userId);
          return socket.emit("error", { 
            message: "사용자를 찾을 수 없습니다." 
          });
        }
    
        // 채팅방 구독 여부 확인
        const isSubscribed = user.subscriptions.includes(roomId);
        if (!isSubscribed) {
          console.error('구독하지 않은 채팅방:', { userId, roomId });
          return socket.emit("error", { 
            message: "구독하지 않은 채팅방입니다. 먼저 채팅방을 구독해주세요." 
          });
        }
    
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
          console.error('존재하지 않는 방:', roomId);
          return socket.emit("error", { 
            message: "존재하지 않는 방입니다." 
          });
        }
    
        const validation = validateChat(data.content);
        if (!validation.isValid) {
          console.error('메시지 검증 실패:', { content: data.content, message: validation.message });
          return socket.emit("error", { 
            message: validation.message 
          });
        }
        
        const chat = await Chat.create({
          room: roomId,
          user: userId,
          content: validation.safeChat,
          createdAt: new Date(),
        });
        
        await Room.updateOne(
          { _id: roomId },
          { 
            $inc: { totalMessageCount: 1 },
            $set: { lastMessage: chat._id }
          }
        );
        
        // 자신이 보낸 메시지는 읽은 것으로 처리
        await updateReadCount(userId, roomId);
        
        const populatedChat = await Chat.findById(chat._id)
          .populate('user', 'nickname profileUrl email');
        
        io.of("/chat").to(roomId).emit("chat", populatedChat);
        
        // 새로운 메시지가 도착했을 때 현재 채팅방을 보고 있는 모든 사용자의 읽음 상태 업데이트
        updateCurrentViewingReadCount(roomId, io);
        
      } catch (error) {
        console.error('메시지 처리 오류:', error);
        socket.emit("error", { 
          message: "메시지 저장 중 오류가 발생했습니다.",
          details: error.message
        });
      }
    });

    socket.on("disconnect", async () => {
      
      try {
        const rooms = Array.from(socket.rooms);
        
        const chatRooms = rooms.filter(room => room !== socket.id);
        
        // 각 방에서 퇴장 처리
        for (const roomId of chatRooms) {
          if (!roomId) continue;

          const room = await Room.findById(roomId)
            .populate({
              path: 'lastMessage',
              select: 'content createdAt',
              populate: {
                path: 'user',
                select: 'nickname profileUrl'
              }
            });
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

// 메시지가 왔을 때 현재 채팅방을 보고 있는 모든 사용자의 읽음 상태 업데이트
async function updateCurrentViewingReadCount(roomId, io) {
  try {
    const sockets = await io.of('/chat').in(roomId).fetchSockets();
    
    for (const socket of sockets) {
      if (socket.user && socket.currentViewingRoom === roomId) {
        await updateReadCount(socket.user.id, roomId);
      }
    }
  } catch (error) {
    console.error('읽음 상태 업데이트 오류:', error);
  }
}

async function updateReadCount(userId, roomId) {
  try {
    const room = await Room.findById(roomId)
      .populate({
        path: 'lastMessage',
        select: 'content createdAt',
        populate: {
          path: 'user',
          select: 'nickname profileUrl'
        }
      });
    if (!room) return;
    

    const totalMessageCount = room.totalMessageCount;
    

    const user = await User.findById(userId);
    if (!user) return;
    
    if (!user.readCounts) {
      await User.updateOne(
        { _id: userId },
        { $set: { readCounts: {} } }
      );
    }
    
    const currentReadCount = user.readCounts && user.readCounts[roomId] 
      ? user.readCounts[roomId] 
      : 0;
    
    if (currentReadCount >= totalMessageCount) {
      return;
    }
    
    // 객체 형태로 readCounts 업데이트
    // MongoDB에서 동적 키를 사용하기 위한 표현식
    const updateQuery = {};
    updateQuery[`readCounts.${roomId}`] = totalMessageCount;
    
    await User.updateOne(
      { _id: userId },
      { $set: updateQuery }
    );
    
  } catch (error) {
    console.error('읽음 상태 업데이트 오류:', error);
  }
}
