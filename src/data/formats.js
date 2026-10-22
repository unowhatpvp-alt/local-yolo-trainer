import {zip,unzip,strToU8,strFromU8} from 'fflate';
import {sha256,validateProject,validateClasses,uid} from './project.js';
import {validateCheckpoint as validatePortableCheckpoint} from '../model/checkpoint-schema.js';

const MAX_ARCHIVE=512*1024*1024,MAX_EXPANDED=768*1024*1024,MAX_FILES=4100,MAX_MODEL=16*1024*1024;
const zipFiles=files=>new Promise((resolve,reject)=>zip(files,{level:0},(error,data)=>error?reject(error):resolve(data)));
const encode=value=>strToU8(JSON.stringify(value));
function safePath(name){return typeof name==='string'&&name.length<300&&!name.startsWith('/')&&!name.includes('\\')&&!name.split('/').some(x=>['..','.','__proto__','prototype','constructor',''].includes(x))&&!name.includes('\0');}
async function unpack(blob,{model=false}={}){
 if(!(blob instanceof Blob)||blob.size<4||blob.size>(model?MAX_MODEL+4*1024*1024:MAX_ARCHIVE))throw new Error('ARCHIVE_SIZE');
 const data=new Uint8Array(await blob.arrayBuffer());let count=0,total=0,bad=false;
 const result=await new Promise((resolve,reject)=>unzip(data,{filter:file=>{count++;total+=file.originalSize;if(!safePath(file.name)||count>(model?2:MAX_FILES)||file.originalSize>(model?MAX_MODEL:MAX_ARCHIVE)||total>(model?MAX_MODEL+4*1024*1024:MAX_EXPANDED)||(model&&!['manifest.json','weights.bin'].includes(file.name))){bad=true;return false;}return true;}},(error,files)=>error?reject(error):resolve(files)));
 if(bad)throw new Error('ARCHIVE_UNSAFE');
 return result;
}
function readJson(files,name,max=4*1024*1024){const data=files[name];if(!data||data.length>max)throw new Error('ARCHIVE_MANIFEST');return JSON.parse(strFromU8(data));}

export function validateCheckpoint(checkpoint){
 validatePortableCheckpoint(checkpoint);
 if(!checkpoint||checkpoint.schemaVersion!==1||!['yolo26n-frozen-head-v1','tiny-grid-v1'].includes(checkpoint.architecture)||checkpoint.resolution!==128||checkpoint.slots!==2)throw new Error('CHECKPOINT_ARCHITECTURE');
 validateClasses(checkpoint.classes);
 if(!Array.isArray(checkpoint.weights)||checkpoint.weights.length<4||checkpoint.weights.length>16)throw new Error('CHECKPOINT_WEIGHTS');
 let total=0;
 for(const list of [checkpoint.weights,checkpoint.optimizer?.weights||[]]){
  const names=new Set();if(list.length>40)throw new Error('CHECKPOINT_WEIGHTS');
  for(const weight of list){
   if(!weight||typeof weight.name!=='string'||!weight.name||weight.name.length>150||names.has(weight.name))throw new Error('CHECKPOINT_WEIGHT_NAME');names.add(weight.name);
   if(!Array.isArray(weight.shape)||weight.shape.length>4||!weight.shape.every(x=>Number.isInteger(x)&&x>0&&x<100000))throw new Error('CHECKPOINT_SHAPE');
   const length=weight.shape.reduce((a,b)=>a*b,1);
   if(!Number.isSafeInteger(length)||length>MAX_MODEL/4||!['float32','int32'].includes(weight.dtype)||!ArrayBuffer.isView(weight.data)||weight.data.length!==length||![Float32Array,Int32Array].some(Type=>weight.data instanceof Type))throw new Error('CHECKPOINT_DATA');
   if(weight.dtype==='float32'&&!(weight.data instanceof Float32Array)||weight.dtype==='int32'&&!(weight.data instanceof Int32Array))throw new Error('CHECKPOINT_DTYPE');
   for(const value of weight.data)if(!Number.isFinite(value))throw new Error('CHECKPOINT_NONFINITE');
   total+=weight.data.byteLength;if(total>MAX_MODEL)throw new Error('CHECKPOINT_SIZE');
  }
 }
 if(!checkpoint.training||!Number.isInteger(checkpoint.training.step)||checkpoint.training.step<0)throw new Error('CHECKPOINT_TRAINING_STATE');
 return checkpoint;
}
async function encodeCheckpoint(checkpoint){
 validateCheckpoint(checkpoint);const parts=[];let offset=0;
 const specs=list=>list.map(w=>{const bytes=new Uint8Array(w.data.buffer,w.data.byteOffset,w.data.byteLength);parts.push(bytes);const spec={name:w.name,shape:w.shape,dtype:w.dtype,offset,byteLength:bytes.length};offset+=bytes.length;return spec;});
 const manifest={...checkpoint,format:'local-yolo-trainer-model',weights:specs(checkpoint.weights),optimizer:checkpoint.optimizer?{...checkpoint.optimizer,weights:specs(checkpoint.optimizer.weights||[])}:undefined};
 const bytes=new Uint8Array(offset);let position=0;for(const part of parts){bytes.set(part,position);position+=part.length;}
 manifest.payloadSha256=await sha256(bytes);
 return {'manifest.json':encode(manifest),'weights.bin':bytes};
}
async function decodeCheckpoint(files){
 const manifest=readJson(files,'manifest.json'),bytes=files['weights.bin'];
 if(manifest.format!=='local-yolo-trainer-model'||!bytes||bytes.length>MAX_MODEL||await sha256(bytes)!==manifest.payloadSha256)throw new Error('CHECKPOINT_INTEGRITY');
 const occupied=[];
 const decode=list=>{if(!Array.isArray(list)||list.length>40)throw new Error('CHECKPOINT_WEIGHTS');return list.map(w=>{
  if(!Number.isInteger(w.offset)||!Number.isInteger(w.byteLength)||w.offset<0||w.byteLength<0||w.offset%4||w.byteLength%4||w.offset+w.byteLength>bytes.length||occupied.some(([start,end])=>w.offset<end&&w.offset+w.byteLength>start))throw new Error('CHECKPOINT_OFFSETS');
  occupied.push([w.offset,w.offset+w.byteLength]);const buffer=bytes.slice(w.offset,w.offset+w.byteLength).buffer,Type=w.dtype==='int32'?Int32Array:Float32Array;
  return {name:w.name,shape:w.shape,dtype:w.dtype,data:new Type(buffer)};
 });};
 const checkpoint={...manifest,weights:decode(manifest.weights),optimizer:manifest.optimizer?{...manifest.optimizer,weights:decode(manifest.optimizer.weights)}:undefined};
 delete checkpoint.format;delete checkpoint.payloadSha256;return validateCheckpoint(checkpoint);
}
export async function packCheckpoint(checkpoint){return new Blob([await zipFiles(await encodeCheckpoint(checkpoint))],{type:'application/zip'});}
export async function unpackCheckpoint(blob){return decodeCheckpoint(await unpack(blob,{model:true}));}

export async function packProject(project,getBlob,checkpoint){
 validateProject(project);const files={},entries=[];let total=0;
 for(let n=0;n<project.images.length;n++){
  const image=project.images[n],blob=await getBlob(image.id);
  if(!(blob instanceof Blob)||await sha256(blob)!==image.sha256)throw new Error('PROJECT_IMAGE_INTEGRITY');
  total+=blob.size;if(total>MAX_ARCHIVE-8*1024*1024)throw new Error('PROJECT_EXPORT_SIZE');
  const name=`images/${n}.${image.mime==='image/png'?'png':image.mime==='image/webp'?'webp':'jpg'}`;
  files[name]=new Uint8Array(await blob.arrayBuffer());entries.push({id:image.id,path:name,sha256:image.sha256,size:blob.size});
 }
 let checkpointPath=null;
 if(checkpoint){const model=await packCheckpoint(checkpoint);checkpointPath='model.lymodel';files[checkpointPath]=new Uint8Array(await model.arrayBuffer());}
 const manifestBytes=encode({format:'local-yolo-trainer-project',schemaVersion:1,project,images:entries,checkpointPath});
 if(manifestBytes.length>4*1024*1024)throw new Error('PROJECT_METADATA_SIZE');files['manifest.json']=manifestBytes;
 return new Blob([await zipFiles(files)],{type:'application/zip'});
}
export async function unpackProject(blob){
 const files=await unpack(blob),manifest=readJson(files,'manifest.json');
 if(manifest.format!=='local-yolo-trainer-project'||manifest.schemaVersion!==1)throw new Error('PROJECT_ARCHIVE_VERSION');
 const project=validateProject(manifest.project);
 if(!Array.isArray(manifest.images)||manifest.images.length!==project.images.length)throw new Error('PROJECT_IMAGES');
 const images=new Map(),mapping=new Map();
 for(const entry of manifest.images){
  const image=project.images.find(i=>i.id===entry.id),data=files[entry.path];
  if(!image||mapping.has(entry.id)||!safePath(entry.path)||!data||data.length!==entry.size||entry.sha256!==image.sha256||await sha256(data)!==entry.sha256)throw new Error('PROJECT_IMAGE_INTEGRITY');
  const id=uid();mapping.set(entry.id,id);images.set(id,new Blob([data],{type:image.mime}));
 }
 const checkpoint=manifest.checkpointPath?await unpackCheckpoint(new Blob([files[manifest.checkpointPath]||new Uint8Array()])):null;
 project.id=uid();project.title=project.title.slice(0,90);project.updatedAt=new Date().toISOString();
 for(const image of project.images)image.id=mapping.get(image.id);
 if(checkpoint&&(checkpoint.classes.length!==project.classes.length||checkpoint.classes.some((name,i)=>name!==project.classes[i])))throw new Error('PROJECT_MODEL_CLASSES');
 return {project,images,checkpoint};
}
export async function exportYoloDataset(project,getBlob){
 validateProject(project);const reviewed=project.images.filter(i=>i.reviewed);if(!reviewed.length)throw new Error('NO_REVIEWED_IMAGES');
 const files={},names=project.classes.map(name=>`  - ${JSON.stringify(name)}`).join('\n');
 files['data.yaml']=strToU8(`path: .\ntrain: images/train\nval: images/val\nnames:\n${names}\n`);
 for(let index=0;index<reviewed.length;index++){
  const image=reviewed[index],split=image.split==='validation'?'val':'train',stem=String(index).padStart(6,'0'),ext=image.mime==='image/png'?'png':image.mime==='image/webp'?'webp':'jpg',blob=await getBlob(image.id);
  if(!(blob instanceof Blob)||await sha256(blob)!==image.sha256)throw new Error('PROJECT_IMAGE_INTEGRITY');
  files[`images/${split}/${stem}.${ext}`]=new Uint8Array(await blob.arrayBuffer());
  files[`labels/${split}/${stem}.txt`]=strToU8(image.boxes.map(b=>`${b.classId} ${((b.x+b.width/2)/image.width).toFixed(8)} ${((b.y+b.height/2)/image.height).toFixed(8)} ${(b.width/image.width).toFixed(8)} ${(b.height/image.height).toFixed(8)}`).join('\n')+(image.boxes.length?'\n':''));
 }
 files['README.txt']=strToU8('Oriented original-resolution pixels, YOLO normalized xywh labels. Empty label files are reviewed negative images. Image groups are split before training; identical images remain in one group. Validation metrics with very small datasets are not reliable generalization estimates.\n');
 return new Blob([await zipFiles(files)],{type:'application/zip'});
}
