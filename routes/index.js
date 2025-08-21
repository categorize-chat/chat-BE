const express = require('express');
const cors = require('cors');
const { authMiddleware } = require('../middlewares/auth');
const { loginUser, logoutUser } = require('../controllers/auth');
const { registerLocalUser } = require('../controllers/auth');
const upload = require('../middlewares/uploadMiddleware');
const profileController = require('../controllers/profileController');
const {
  registerUser, renderMain, createRoom, enterRoom, sendChat, classifyChat, 
  searchRooms, subscribeRoom, unsubscribeRoom, getRooms, getUserSettings, updateUserNickname, getUnreadCount
} = require('../controllers');

const router = express.Router();

router.use(cors({credentials: true, origin: '*',}));

router.post('/user/login', loginUser);

router.post('/user/join', registerLocalUser);

router.post('/user/logout', logoutUser);

router.post('/user', registerUser);

router.get('/chat', authMiddleware, renderMain);

router.get('/search', authMiddleware, getRooms);

router.post('/search', authMiddleware, searchRooms);

router.post('/subscribe/:roomId', authMiddleware, subscribeRoom);

router.post('/unsubscribe/:roomId', authMiddleware, unsubscribeRoom);

router.post('/chat', authMiddleware, createRoom);

router.post('/chat/summary', authMiddleware, classifyChat);

router.get('/chat/:id', authMiddleware, enterRoom);

router.post('/chat/:id', authMiddleware, sendChat);

router.get('/settings', authMiddleware, getUserSettings);

router.post('/settings/nickname-change', authMiddleware, updateUserNickname);

router.get('/unread', authMiddleware, getUnreadCount);

router.post('/settings/profile-image', 
  authMiddleware,
  upload.single('profileImage'),
  profileController.updateProfileImage
);

module.exports = router;
