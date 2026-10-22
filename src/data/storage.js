import {validateProject,sha256} from './project.js';
import {validateCheckpoint} from './formats.js';
const DATABASE='local-yolo-trainer-v1';
let opening;
const clone=value=>structuredClone(value);
export function openDatabase(){
 if(opening)return opening;
 opening=new Promise((resolve,reject)=>{
  if(!globalThis.indexedDB){reject(new Error('STORAGE_UNAVAILABLE'));return;}
  const request=indexedDB.open(DATABASE,1);
  request.onupgradeneeded=()=>{const db=request.result;db.createObjectStore('projects',{keyPath:'id'});db.createObjectStore('images');db.createObjectStore('thumbnails');db.createObjectStore('checkpoints');};
  request.onerror=()=>reject(request.error||new Error('STORAGE_OPEN_FAILED'));
  request.onblocked=()=>reject(new Error('STORAGE_BLOCKED'));
  request.onsuccess=()=>{const db=request.result;db.onversionchange=()=>{db.close();opening=null;};resolve(db);};
 }).catch(error=>{opening=null;throw error;});
 return opening;
}
async function transact(stores,mode,operation){
 const db=await openDatabase();
 return new Promise((resolve,reject)=>{
  let result;const tx=db.transaction(stores,mode);
  tx.oncomplete=()=>resolve(result);tx.onabort=()=>reject(tx.error||new Error('STORAGE_ABORTED'));tx.onerror=()=>{};
  const fail=error=>{try{tx.abort();}catch{}reject(error);};
  try{operation(tx,value=>{result=value;},fail);}catch(error){fail(error);}
 });
}
function get(store,key){return transact([store],'readonly',(tx,set)=>{const req=tx.objectStore(store).get(key);req.onsuccess=()=>set(req.result);});}
export const getImage=id=>get('images',id);
export const getThumbnail=id=>get('thumbnails',id);
export const getProject=id=>get('projects',id);
export const getCheckpoint=projectId=>get('checkpoints',`${projectId}:current`);
export const getPreviousCheckpoint=projectId=>get('checkpoints',`${projectId}:previous`);
export function listProjects(){return transact(['projects'],'readonly',(tx,set)=>{const req=tx.objectStore('projects').getAll();req.onsuccess=()=>{
 const valid=[],invalidProjects=[];
 for(const record of req.result){try{validateProject(record);valid.push(record);}catch(error){invalidProjects.push({id:typeof record?.id==='string'?record.id:null,reason:error.code||error.message||'PROJECT_INVALID'});}}
 valid.sort((a,b)=>b.updatedAt.localeCompare(a.updatedAt));
 // Quarantine by filtering the list only. Stored bytes remain recoverable and
 // callers can explain skipped records without one bad item blocking startup.
 valid.invalidProjects=invalidProjects;set(valid);
};});}
export function saveProject(project){validateProject(project);const copy=clone(project);return transact(['projects'],'readwrite',tx=>tx.objectStore('projects').put(copy));}
export function saveImage(id,blob,thumbnail){
 if(!(blob instanceof Blob)||!(thumbnail instanceof Blob))throw new Error('IMAGE_BLOB_REQUIRED');
 return transact(['images','thumbnails'],'readwrite',tx=>{tx.objectStore('images').put(blob,id);tx.objectStore('thumbnails').put(thumbnail,id);});
}
/** Never announce a checkpoint saved before the atomic transaction commits. */
export function saveCheckpoint(projectId,checkpoint){
 validateCheckpoint(checkpoint);
 const copy=clone(checkpoint);
 return transact(['checkpoints'],'readwrite',tx=>{
  const store=tx.objectStore('checkpoints'),request=store.get(`${projectId}:current`);
  request.onsuccess=()=>{if(request.result)store.put(request.result,`${projectId}:previous`);store.put(copy,`${projectId}:current`);};
 });
}
/** Import a new ID only. Existing project data is never overwritten by a backup. */
export async function importProjectAtomic(project,images,thumbnails,checkpoint){
 const copy=clone(project),model=checkpoint?clone(checkpoint):null,blobs=new Map(images),thumbs=new Map(thumbnails||[]);
 validateProject(copy);
 if(model){validateCheckpoint(model);if(model.classes.length!==copy.classes.length||model.classes.some((name,index)=>name!==copy.classes[index]))throw new Error('PROJECT_MODEL_CLASSES');}
 if(blobs.size!==copy.images.length||[...blobs.keys()].some(id=>!copy.images.some(image=>image.id===id))||[...thumbs.keys()].some(id=>!blobs.has(id)))throw new Error('PROJECT_IMAGES');
 for(const image of copy.images){const blob=blobs.get(image.id);if(!(blob instanceof Blob)||blob.type!==image.mime||await sha256(blob)!==image.sha256)throw new Error('PROJECT_IMAGE_INTEGRITY');const thumbnail=thumbs.get(image.id);if(thumbnail&&!(thumbnail instanceof Blob))throw new Error('IMAGE_BLOB_REQUIRED');}
 if(await getProject(copy.id))throw new Error('PROJECT_ID_EXISTS');
 return transact(['projects','images','thumbnails','checkpoints'],'readwrite',tx=>{
  tx.objectStore('projects').add(copy);
  for(const image of copy.images){tx.objectStore('images').add(blobs.get(image.id),image.id);const thumb=thumbs.get(image.id);if(thumb)tx.objectStore('thumbnails').add(thumb,image.id);}
  if(model)tx.objectStore('checkpoints').add(model,`${copy.id}:current`);
 });
}
/** Caller has confirmed removal. Only this photo and its thumbnail are deleted. */
export function removeImageConfirmed(nextProject,imageId){
 const copy=clone(nextProject);validateProject(copy);
 if(typeof imageId!=='string'||!imageId||copy.images.some(image=>image.id===imageId))throw new Error('IMAGE_REMOVAL_INVALID');
 return transact(['projects','images','thumbnails'],'readwrite',(tx,set,fail)=>{
  const projects=tx.objectStore('projects'),request=projects.get(copy.id);
  request.onsuccess=()=>{try{
   const previous=request.result;
   if(!previous||!previous.images.some(image=>image.id===imageId))throw new Error('IMAGE_NOT_FOUND');
   const expected={...previous,images:previous.images.filter(image=>image.id!==imageId),updatedAt:null};
   if(JSON.stringify(expected)!==JSON.stringify({...copy,updatedAt:null}))throw new Error('IMAGE_REMOVAL_CONFLICT');
   projects.put(copy);tx.objectStore('images').delete(imageId);tx.objectStore('thumbnails').delete(imageId);set(copy);
  }catch(error){fail(error);}};
 });
}
/** Caller must obtain explicit product confirmation. No background garbage collector. */
export function deleteProjectConfirmed(project){
 return transact(['projects','images','thumbnails','checkpoints'],'readwrite',tx=>{
  tx.objectStore('projects').delete(project.id);
  for(const image of project.images){tx.objectStore('images').delete(image.id);tx.objectStore('thumbnails').delete(image.id);}
  tx.objectStore('checkpoints').delete(`${project.id}:current`);tx.objectStore('checkpoints').delete(`${project.id}:previous`);
 });
}
export async function storageEstimate(){try{const e=await navigator.storage?.estimate();return e?{usage:e.usage,quota:e.quota}:null;}catch{return null;}}

export class SaveQueue{
 constructor(save,onState=()=>{}){this.save=save;this.onState=onState;this.pending=null;this.timer=null;this.running=null;this.error=null;}
 enqueue(value){this.pending=clone(value);this.onState('pending');clearTimeout(this.timer);this.timer=setTimeout(()=>this.flush().catch(()=>{}),300);}
 async flush(){
  clearTimeout(this.timer);this.timer=null;
  if(this.running){await this.running;if(this.pending)return this.flush();return;}
  if(!this.pending)return;
  const value=this.pending;this.pending=null;this.onState('saving');
  // Begin in a microtask so synchronous failures cannot clear running before
  // the assignment, leaving a rejected promise that blocks every retry.
  this.running=Promise.resolve().then(async()=>{try{await this.save(value);this.error=null;this.onState(this.pending?'pending':'saved');}catch(error){if(!this.pending)this.pending=value;this.error=error;this.onState('error',error);throw error;}finally{this.running=null;}});
  await this.running;
  if(this.pending)return this.flush();
 }
 get dirty(){return Boolean(this.pending||this.running);}
}
