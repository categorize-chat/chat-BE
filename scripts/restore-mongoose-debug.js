const fs = require('fs');
const path = require('path');

const schemasIndexPath = path.join(__dirname, '../schemas/index.js');
const backupPath = path.join(__dirname, '../schemas/index.js.bak');

if (fs.existsSync(backupPath)) {
  console.log('백업 파일에서 원본 복원 중...');
  fs.copyFileSync(backupPath, schemasIndexPath);
  console.log('schemas/index.js 파일이 원본으로 복원되었습니다.');
} else {
  console.error('백업 파일을 찾을 수 없습니다:', backupPath);
  process.exit(1);
} 