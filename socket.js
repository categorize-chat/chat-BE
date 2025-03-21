const SocketIO = require("socket.io");
const Room = require("./schemas/room");
const Chat = require("./schemas/chat");
const User = require("./schemas/user");
const { verifyToken } = require('./utils/jwt');

function validateAndSanitizeChat(content) {
  if (!content || typeof content !== 'string') {
    return { 
      isValid: false, 
      message: "올바른 채팅 내용을 입력해주세요. 문자열 형태여야 합니다." 
    };
  }

  const trimmedContent = content.trim();
  
  if (trimmedContent.length === 0) {
    return { 
      isValid: false, 
      message: "채팅 내용을 입력해주세요." 
    };
  }
  
  if (trimmedContent.length > 1000) {
    return { 
      isValid: false, 
      message: "채팅은 1000자 이하여야 합니다." 
    };
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
    console.log("chat 네임스페이스에 접속:", socket.id, socket.user?.nickname);
    
    // 에러 이벤트 리스너 추가
    socket.on("error", (error) => {
      console.error("소켓 에러:", error);
    });
    
    // 현재 사용자가 보고 있는 채팅방 ID를 저장
    let currentViewingRoom = null;
    
    socket.on("join", async (data) => {
      data.forEach(roomId => {  
        console.log("채팅방 입장:", roomId, socket.user?.nickname);
        socket.join(roomId);
      });

      try {
        // TODO: 채팅방에 유저 추가
        // const room = await Room.findById(data);
        // if (room && !room.participants.includes(socket.user.id)) {
        //   room.participants.push(socket.user.id);
        //   await room.save();
        // }
      } catch (error) {
        console.error('Room join error:', error);
      }
    });

    // 사용자가 채팅방 화면을 보고 있을 때 읽음 상태 업데이트
    socket.on("view", async (roomId) => {
        currentViewingRoom = roomId;
      try {
        console.log(`사용자 ${socket.user.nickname}(${socket.user.id})가 채팅방 ${roomId}를 보고 있습니다.`);
        
        // 채팅방을 보고 있으므로 모든 메시지를 읽은 것으로 처리
        await updateUserReadCount(socket.user.id, roomId);
      } catch (error) {
        console.error('채팅방 읽음 상태 업데이트 오류:', error);
      }
    });
    
    // 사용자가 채팅방 화면을 벗어날 때
    socket.on("leave", () => {
      currentViewingRoom = null;
    });

    socket.on("message", async (data) => {
      console.log('메시지 수신:', data);
      
      try {
        // 필수 데이터 확인
        if (!data || !data.room || !data.content) {
          console.error('잘못된 메시지 형식:', data);
          return socket.emit("error", { 
            message: "메시지 형식이 올바르지 않습니다. room과 content가 필요합니다." 
          });
        }
        
        const userId = socket.user.id;
        const roomId = data.room;
    
        // 사용자 구독 정보 확인
        const user = await User.findById(userId).select('subscriptions');
        if (!user) {
          console.error('사용자를 찾을 수 없음:', userId);
          return socket.emit("error", { 
            message: "사용자를 찾을 수 없습니다." 
          });
        }
    
        // 구독 여부 확인
        const isSubscribed = user.subscriptions.includes(roomId);
        if (!isSubscribed) {
          console.error('구독하지 않은 채팅방:', { userId, roomId });
          return socket.emit("error", { 
            message: "구독하지 않은 채팅방입니다. 먼저 채팅방을 구독해주세요." 
          });
        }
    
        // 방 존재 여부 확인
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
    
        // 메시지 검증
        const validation = validateAndSanitizeChat(data.content);
        if (!validation.isValid) {
          console.error('메시지 검증 실패:', { content: data.content, message: validation.message });
          return socket.emit("error", { 
            message: validation.message 
          });
        }
    
        console.log('메시지 저장 시작:', { roomId, userId, content: validation.sanitizedContent });
        
        // 검증된 내용으로 채팅 생성
        const chat = await Chat.create({
          room: roomId,
          user: userId,
          content: validation.sanitizedContent,
          createdAt: new Date(),
        });
        
        console.log('채팅 생성 완료:', chat._id);
        
        // 1. Room의 totalMessageCount 증가 및 lastMessage 업데이트 (원자적 연산)
        await Room.updateOne(
          { _id: roomId },
          { 
            $inc: { totalMessageCount: 1 },
            $set: { lastMessage: chat._id }
          }
        );
        
        // 2. 메시지 전송자의 읽음 상태 업데이트 (자신이 보낸 메시지는 읽은 것으로 처리)
        await updateUserReadCount(userId, roomId);
        
        // populate로 user 정보를 포함하여 조회
        const populatedChat = await Chat.findById(chat._id)
          .populate('user', 'nickname profileUrl email');
        
        console.log('채팅 발송 준비 완료:', { chatId: chat._id, roomId });
        
        // 채팅 발송
        io.of("/chat").to(roomId).emit("chat", populatedChat);
        
        // 새로운 메시지가 도착했을 때 현재 채팅방을 보고 있는 모든 사용자의 읽음 상태 업데이트
        updateActiveViewersReadCount(roomId, io);
        
        console.log('채팅 발송 완료');
      } catch (error) {
        console.error('메시지 처리 오류:', error);
        socket.emit("error", { 
          message: "메시지 저장 중 오류가 발생했습니다.",
          details: error.message
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

// 새로운 함수: 현재 채팅방을 보고 있는 모든 사용자의 읽음 상태 업데이트
async function updateActiveViewersReadCount(roomId, io) {
  try {
    // 해당 채팅방의 소켓 목록 가져오기
    const sockets = await io.of('/chat').in(roomId).fetchSockets();
    
    for (const socket of sockets) {
      // 소켓에 사용자 정보가 있고, 현재 보고 있는 채팅방이 해당 roomId와 일치하는 경우
      if (socket.user && socket.currentViewingRoom === roomId) {
        await updateUserReadCount(socket.user.id, roomId);
      }
    }
  } catch (error) {
    console.error('활성 사용자 읽음 상태 업데이트 오류:', error);
  }
}

// 사용자의 읽음 상태 업데이트 함수
async function updateUserReadCount(userId, roomId) {
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
    
    // 현재 방의 총 메시지 수 가져오기
    const totalMessageCount = room.totalMessageCount;
    
    // 사용자 정보 가져오기
    const user = await User.findById(userId);
    if (!user) return;
    
    // readCounts가 없으면 초기화
    if (!user.readCounts) {
      await User.updateOne(
        { _id: userId },
        { $set: { readCounts: {} } }
      );
    }
    
    // 현재 사용자의 해당 방에 대한 읽음 상태
    const currentReadCount = user.readCounts && user.readCounts[roomId] 
      ? user.readCounts[roomId] 
      : 0;
    
    // 사용자가 이미 최신 상태라면 업데이트하지 않음
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
    
    // 간결한 디버깅 로그
    console.log(`읽음 상태 업데이트: 사용자=${userId}, 방=${roomId}, 읽은 메시지 수=${totalMessageCount}`);
  } catch (error) {
    console.error('읽음 상태 업데이트 오류:', error);
  }
}
