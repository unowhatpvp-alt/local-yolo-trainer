import {sha256,uid} from './project.js';
const MAX_FILE=32*1024*1024,MAX_PIXELS=32_000_000;
export async function decodeImage(blob){
 if(!(blob instanceof Blob)||blob.size===0||blob.size>MAX_FILE)throw new Error('IMAGE_FILE_TOO_LARGE');
 if(typeof createImageBitmap==='function')return createImageBitmap(blob,{imageOrientation:'from-image'});
 return new Promise((resolve,reject)=>{const image=new Image(),url=URL.createObjectURL(blob);image.onload=()=>{URL.revokeObjectURL(url);resolve(image);};image.onerror=()=>{URL.revokeObjectURL(url);reject(new Error('IMAGE_DECODE_FAILED'));};image.src=url;});
}
function canvas(width,height){const c=document.createElement('canvas');c.width=width;c.height=height;return c;}
function toBlob(c,type='image/jpeg',quality=0.94){return new Promise((resolve,reject)=>c.toBlob(blob=>blob?resolve(blob):reject(new Error('IMAGE_ENCODE_FAILED')),type,quality));}
async function thumbnailFromImage(image,size){const width=image.width||image.naturalWidth,height=image.height||image.naturalHeight,scale=Math.min(1,size/Math.max(width,height)),c=canvas(Math.max(1,Math.round(width*scale)),Math.max(1,Math.round(height*scale)));c.getContext('2d').drawImage(image,0,0,c.width,c.height);const result=await toBlob(c,'image/webp',0.8);c.width=c.height=1;return result;}
export async function makeThumbnail(blob,size=256){
 const image=await decodeImage(blob);
 try{return await thumbnailFromImage(image,size);}finally{image.close?.();}
}
/** Canonical oriented pixels strip EXIF and never modify the user's original file. */
export async function importImage(file){
 if(file.size>MAX_FILE)throw new Error('IMAGE_FILE_TOO_LARGE');
 const sourceHash=await sha256(file),image=await decodeImage(file);
 try{
  const width=image.width||image.naturalWidth,height=image.height||image.naturalHeight;
  if(!width||!height||width*height>MAX_PIXELS)throw new Error('IMAGE_PIXELS_TOO_LARGE');
  const c=canvas(width,height),ctx=c.getContext('2d');ctx.drawImage(image,0,0);
  const type=file.type==='image/png'?'image/png':'image/jpeg';
  const blob=await toBlob(c,type,0.95);c.width=c.height=1;
  const thumbnail=await thumbnailFromImage(image,256),hash=await sha256(blob);
  return {image:{id:uid(),name:String(file.name||'image').slice(0,250),mime:blob.type,width,height,sha256:hash,sourceSha256:sourceHash,boxes:[],reviewed:false},blob,thumbnail};
 }finally{image.close?.();}
}
export function downloadBlob(blob,filename){
 const url=URL.createObjectURL(blob),link=document.createElement('a');link.href=url;link.download=filename;link.hidden=true;document.body.append(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(url),60_000);
}
