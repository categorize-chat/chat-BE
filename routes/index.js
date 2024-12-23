const express = require('express');
const cors = require('cors');
const { authMiddleware } = require('../middlewares/auth');

const {
  registerUser, renderMain, createRoom, enterRoom, sendChat, classifyChat, searchRooms
} = require('../controllers');

const router = express.Router();

router.use(cors({credentials: true, origin: '*',}));

router.post('/user', registerUser);

router.get('/chat', authMiddleware, renderMain);

router.get('/search', authMiddleware, searchRooms);

router.post('/subscribe/:roomId', authMiddleware, subscribeRoom);

router.post('/chat', authMiddleware, createRoom);

router.post('/chat/summary', authMiddleware, classifyChat);

router.get('/chat/:id', authMiddleware, enterRoom);

router.post('/chat/:id', authMiddleware, sendChat);

module.exports = router;