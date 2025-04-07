# 프론트엔드 프록시 설정 가이드

## 개요

채팅 분류 기능을 로컬에서 실행하기 위해, 프론트엔드의 API 요청 중 주제 요약 관련 요청만 로컬 API 서버로 리다이렉트하는 방법을 설명합니다.

## 방법 1: 개발 환경에서 Vite 프록시 설정 사용

chat-FE 프로젝트를 로컬에서 실행하는 경우, vite.config.ts 파일을 수정하여 API 요청을 리다이렉트할 수 있습니다.

```typescript
import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';
import svgr from '@svgr/rollup';
import tsconfigPaths from 'vite-tsconfig-paths';

// https://vitejs.dev/config/
export default ({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');

  return defineConfig({
    define: {
      __APP_ENV__: JSON.stringify(env.APP_ENV),
    },
    plugins: [react(), svgr(), tsconfigPaths()],
    server: {
      port: +env.VITE_CLIENT_PORT,
      proxy: {
        '/api': {
          target: `${env.VITE_SERVER_URL}`,
          changeOrigin: true,
          secure: false,
          rewrite: path => path.replace(/^\/api/, ''),
        },
        // 주제 요약 API만 로컬 API 서버로 리다이렉트
        '/chat/summary': {
          target: 'http://localhost:3000',
          changeOrigin: true,
          secure: false,
          priority: 100, // 높은 우선순위 설정
        },
      },
    },
  });
};
```

## 방법 2: 배포된 프론트엔드에서 브라우저 확장 프로그램 사용

이미 배포된 프론트엔드 애플리케이션을 사용하는 경우, 브라우저 확장 프로그램을 사용하여 특정 요청을 리다이렉트할 수 있습니다.

### Chrome: Requestly 확장 프로그램 사용하기

1. [Requestly 확장 프로그램](https://chrome.google.com/webstore/detail/requestly-redirect-url-mo/mdnleldcmiljblolnjhpnblkcekpdkpa)을 설치합니다.
2. 새로운 리다이렉트 규칙을 생성합니다:
   - 규칙 유형: "Redirect Request"
   - 소스 URL: `https://chat.travaa.site/chat/summary`
   - 대상 URL: `http://localhost:3000/chat/summary`
   - 요청 방법: "POST"

### Firefox: Redirector 확장 프로그램 사용하기

1. [Redirector 확장 프로그램](https://addons.mozilla.org/en-US/firefox/addon/redirector/)을 설치합니다.
2. 새로운 리다이렉트 규칙을 생성합니다:
   - 패턴: `https://chat.travaa.site/chat/summary`
   - 리다이렉트 대상: `http://localhost:3000/chat/summary`
   - 패턴 유형: "정확히 일치"

## 방법 3: API 주소 환경 변수 사용

프론트엔드 코드가 환경 변수를 통해 API 주소를 구성하는 경우, `.env.local` 파일을 생성하여 주제 요약 API 주소만 변경할 수 있습니다:

```
VITE_SUMMARY_API_URL=http://localhost:3000/chat/summary
```

그리고 프론트엔드 코드에서 다음과 같이 사용합니다:

```typescript
// api/ai/query.ts
export const AiSummaryQuery = () => ({
  mutationKey: [`AI summary`],
  mutationFn: async (req: TAiSummaryRequest) => {
    // 환경 변수 사용 (기본값으로 원래 주소 사용)
    const apiUrl = import.meta.env.VITE_SUMMARY_API_URL || '/chat/summary';
    
    const response = await API.json
      .post(apiUrl, req)
      .then(res => res.data as TAiSummaryResponse)
      .then(({ code, message, result }) => {
        if (code !== 200) {
          throw new Error(message);
        }

        return result;
      });

    return response;
  },
  refetchOnWindowFocus: false,
}); 