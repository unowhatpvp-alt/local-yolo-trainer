import {defineConfig} from 'vite';
export default defineConfig({base:process.env.TRAINER_BASE||'/local-yolo-trainer/',worker:{format:'es'},build:{target:'es2022',chunkSizeWarningLimit:1500},server:{host:'0.0.0.0'}});
