```bash
huggingface-cli download \
  --repo-type dataset \
  svjack/Genshin-Impact-Portrait-with-Tags-Filtered-IID-Gender-SP \
  --local-dir "./genshin_collei_data" \
  --include "genshin_impact_COLLEI_images_and_texts/*" \
  --local-dir-use-symlinks False 

cd ai-toolkit/ui
package.json

"start": "next start -H 0.0.0.0 -p 8675"

cd ai-toolkit

run.py 
run_modal.py 

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

cd ai-toolkit/ui
'''

- 去除字体引用 ai-toolkit/ui/src/app/layout.tsx

'''
import type { Metadata } from 'next';
//import { Inter } from 'next/font/google';
import './globals.css';
import Sidebar from '@/components/Sidebar';
import { ThemeProvider } from '@/components/ThemeProvider';
import ConfirmModal from '@/components/ConfirmModal';
import { Suspense } from 'react';
import AuthWrapper from '@/components/AuthWrapper';
import DocModal from '@/components/DocModal';

export const dynamic = 'force-dynamic';

//const inter = Inter({ subsets: ['latin'] });
const inter = { className: 'font-sans' }

export const metadata: Metadata = {
  title: 'Ostris - AI Toolkit',
  description: 'A toolkit for building AI things.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  // Check if the AI_TOOLKIT_AUTH environment variable is set
  const authRequired = process.env.AI_TOOLKIT_AUTH ? true : false;

  return (
    <html lang="en" className="dark">
      <head>
        <meta name="apple-mobile-web-app-title" content="AI-Toolkit" />
      </head>
      <body className={inter.className}>
        <ThemeProvider>
          <AuthWrapper authRequired={authRequired}>
            <div className="flex h-screen bg-gray-950">
              <Sidebar />
              <main className="flex-1 overflow-auto bg-gray-950 text-gray-100 relative">
                <Suspense>{children}</Suspense>
              </main>
            </div>
          </AuthWrapper>
        </ThemeProvider>
        <ConfirmModal />
        <DocModal />
      </body>
    </html>
  );
}

'''


'''
npm run build_and_start

huggingface-cli download ai-toolkit/Wan2.2-T2V-A14B-Diffusers-bf16 --local-dir="Wan2.2-T2V-A14B-Diffusers-bf16"
```

- Filter Video
```python
import os
import shutil
from tqdm import tqdm
from moviepy.editor import VideoFileClip

def check_and_copy_files(source_dir, target_dir, min_frames=9):
    # 确保目标目录存在
    os.makedirs(target_dir, exist_ok=True)
    
    # 收集所有.mp4文件
    mp4_files = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.mp4'):
                mp4_files.append(os.path.join(root, file))
    
    # 遍历所有.mp4文件，检查帧数并处理
    for mp4_path in tqdm(mp4_files, desc="Processing videos"):
        # 获取对应的.txt文件路径
        txt_path = mp4_path.replace('.mp4', '.txt')
        
        # 检查.txt文件是否存在
        if not os.path.exists(txt_path):
            print(f"Missing .txt file for: {mp4_path}")
            continue
        
        try:
            # 使用moviepy获取视频帧数
            with VideoFileClip(mp4_path) as video:
                frame_count = int(video.fps * video.duration)
            
            # 检查帧数是否满足条件
            if frame_count >= min_frames:
                # 构建目标路径
                rel_path = os.path.relpath(mp4_path, source_dir)
                target_mp4 = os.path.join(target_dir, rel_path)
                target_txt = target_mp4.replace('.mp4', '.txt')
                
                # 确保目标子目录存在
                os.makedirs(os.path.dirname(target_mp4), exist_ok=True)
                
                # 拷贝文件对
                shutil.copy2(mp4_path, target_mp4)
                shutil.copy2(txt_path, target_txt)
            else:
                print(f"Frame count too low: {mp4_path} (frames: {frame_count})")
                
        except Exception as e:
            print(f"Error processing {mp4_path}: {str(e)}")

# 使用示例
source_dir = "Ineffa_videos_captioned_960x544x4"
target_dir = "Ineffa_videos_captioned_960x544x4_9"
check_and_copy_files(source_dir, target_dir)
```

- 14B video config
```txt
resolution: 512
frame_count: 16
disable inference sample
```
