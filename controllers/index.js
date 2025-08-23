const Room = require("../schemas/room");
const Chat = require("../schemas/chat");
const User = require("../schemas/user");
const classifyTopics = require('../services/chatClassifier');
const mongoose = require('mongoose');
const sanitizeHtml = require('sanitize-html');
const { validateChat } = require('../socket');

exports.registerUser = async (req, res, next) => { 
  try {
    const exist = await User.findOne({ nickname: req.body.nickname });

    if (!exist) {
      const newUser = await User.create({
        nickname: req.body.nickname,
      });
      res.json({
        isSuccess: true,
        code: 200,
        message: "요청에 성공했습니다.",
        result: {
          userId: newUser.userId,
          nickname: newUser.nickname,
        },
      });
    } else {
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

// 구독 중인 채팅방 목록 조회
exports.renderRooms = async (req, res) => {
  try {
    const user = await User.findById(req.user.id).populate('subscriptions');
    
    if (!user) {
      return res.status(404).json({
        isSuccess: false,
        code: 404,
        message: "사용자를 찾을 수 없습니다."
      });
    }
    
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

    const channelsWithCounts = rooms.map(room => {
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
    // 프론트와의 협의로 가장 최근 15개 채팅방만 제공하기로 결정
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
    const searchTerm = req.body.search;
    
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

    const escapedSearchTerm = searchTerm.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

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

function validateRoom(channelName, description) {
  if (!channelName || typeof channelName !== 'string') {
    return { isValid: false, message: "올바른 채팅방 이름을 입력해주세요." };
  }

  if (description && typeof description !== 'string') {
    return { isValid: false, message: "올바른 설명을 입력해주세요." };
  }

  const name = channelName.trim();
  const desc = description ? description.trim() : '';
  
  if (name.length < 2 || name.length > 30) {
    return { isValid: false, message: "채팅방 이름은 2-30자 사이여야 합니다." };
  }

  if (desc.length > 200) {
    return { isValid: false, message: "채팅방 설명은 200자를 초과할 수 없습니다." };
  }

  // 허용된 문자만 포함되어 있는지 검사 (알파벳, 숫자, 한글, 일부 특수문자만 허용)
  const nameRegex = /^[a-zA-Z0-9가-힣\s_.-]+$/;
  if (!nameRegex.test(name)) {
    return { isValid: false, message: "채팅방 이름에 허용되지 않은 문자가 포함되어 있습니다." };
  }

  // 채팅방 설명을 통한 XSS 공격 방지
  const safeDesc = sanitizeHtml(desc);

  return { 
    isValid: true, 
    safeName: name, 
    safeDesc 
  };
}

exports.createRoom = async (req, res, next) => {
  try {
    const validation = validateRoom(req.body.channelName, req.body.description);
    if (!validation.isValid) {
      return res.status(400).json({
        isSuccess: false,
        code: 400,
        message: validation.message
      });
    }

    const exist = await Room.findOne({ 
      channelName: validation.safeName 
    });

    if (!exist) {
      try {
        const newRoom = await Room.create({
          channelName: validation.safeName,
          description: validation.safeDesc,
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

exports.enterRoom = async (req, res, next) => {
  try {
    const roomId = req.params.id;
    const userId = req.user.id;
    const cursor = req.query.cursor;
    const limit = parseInt(req.query.limit) || 20;

    const user = await User.findById(userId).select('subscriptions');
    if (!user) {
      return res.status(404).json({
        isSuccess: false,
        code: 404,
        message: "사용자를 찾을 수 없습니다."
      });
    }

    const isSubscribed = user.subscriptions.includes(roomId);
    if (!isSubscribed) {
      return res.status(403).json({
        isSuccess: false,
        code: 403,
        message: "구독하지 않은 채팅방입니다. 먼저 채팅방을 구독해주세요."
      });
    }

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

    const baseQuery = { room: room._id };
    
    // cursor 기반 쿼리 설정
    if (cursor) {
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

      baseQuery.createdAt = { $lt: cursorMessage.createdAt };
    }

    const totalCount = await Chat.countDocuments(baseQuery);

    const messages = await Chat.find(baseQuery)
      .sort({ createdAt: -1 })
      .limit(limit)
      .populate('user', 'nickname profileUrl email');

    const hasNextPage = totalCount > messages.length;
    
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



exports.sendChat = async (req, res, next) => {
  try {
    const roomId = req.params.roomId;
    const userId = req.user.id;

    if (!mongoose.isValidObjectId(roomId)) {
      return res.status(400).json({
        isSuccess: false,
        code: 400,
        message: "유효하지 않은 방 ID입니다."
      });
    }

    const user = await User.findById(userId).select('subscriptions');
    if (!user) {
      return res.status(404).json({
        isSuccess: false,
        code: 404,
        message: "사용자를 찾을 수 없습니다."
      });
    }

    const isSubscribed = user.subscriptions.includes(roomId);
    if (!isSubscribed) {
      return res.status(403).json({
        isSuccess: false,
        code: 403,
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
      return res.status(404).json({
        isSuccess: false,
        code: 404,
        message: "존재하지 않는 방입니다."
      });
    }

    const validation = validateChat(req.body.content);
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
      content: validation.safeChat,
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

exports.classifyChat = async (req, res, next) => {
  try {
    const { channelId, howmany } = req.body;
    const userId = req.user.id;

    if (!channelId) {
      return res.status(400).json({
        isSuccess: false,
        code: 400,
        message: "channelId가 필요합니다.",
      });
    }

    if (!mongoose.isValidObjectId(channelId)) {
      return res.status(400).json({
        isSuccess: false,
        code: 400,
        message: "유효하지 않은 채널 ID입니다."
      });
    }

    const user = await User.findById(userId).select('subscriptions');
    if (!user) {
      return res.status(404).json({
        isSuccess: false,
        code: 404,
        message: "사용자를 찾을 수 없습니다."
      });
    }

    const isSubscribed = user.subscriptions.includes(channelId);
    if (!isSubscribed) {
      return res.status(403).json({
        isSuccess: false,
        code: 403,
        message: "구독하지 않은 채팅방입니다. 먼저 채팅방을 구독해주세요."
      });
    }

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

    // howmany를 정해주지 않았을 경우 채팅 100개만 분류
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

exports.subscribeRoom = async (req, res, next) => {
  try {
    const roomId = req.params.roomId;
    const userId = req.user.id;

    // room의 participants와 user의 subscriptions을 동시에 업데이트
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

function validateNickname(nickname) {
  if (!nickname || typeof nickname !== 'string') {
    return { isValid: false, message: "올바른 닉네임을 입력해주세요." };
  }

  const safeNickname = nickname.trim();
  
  if (safeNickname.length === 0 || safeNickname.length > 15) {
    return { isValid: false, message: "닉네임은 1-15자 사이여야 합니다." };
  }

  const nicknameRegex = /^[a-zA-Z0-9가-힣_.-]+$/;
  if (!nicknameRegex.test(safeNickname)) {
    return { isValid: false, message: "닉네임에 허용되지 않은 문자가 포함되어 있습니다." };
  }

  return { isValid: true, safeNickname };
}


exports.getUserSettings = async (req, res, next) => {
  try {
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


exports.updateUserNickname = async (req, res, next) => {
  try {
    const { nickname } = req.body;


    if (!mongoose.isValidObjectId(req.user.id)) {
      return res.status(400).json({
        isSuccess: false,
        code: 400,
        message: "유효하지 않은 사용자 ID입니다."
      });
    }
    
    const validation = validateNickname(nickname);
    if (!validation.isValid) {
      return res.status(400).json({
        isSuccess: false,
        code: 400,
        message: validation.message
      });
    }

    const updatedUser = await User.findByIdAndUpdate(
      req.user.id,
      { nickname: validation.safeNickname },
      { 
        new: true,
        runValidators: true
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


