const express = require('express');
const cors = require('cors');

const {
  registerUser, renderMain, createRoom, enterRoom, sendChat, classifyChat
} = require('../controllers');

const router = express.Router();

router.use(cors({credentials: true, origin: '*',}));

router.post('/user', registerUser);

router.get('/chat', renderMain);

router.post('/chat', createRoom);

router.post('/chat/summary', classifyChat);

router.get('/chat/:id', enterRoom);

router.post('/chat/:id', sendChat);

module.exports = router;
