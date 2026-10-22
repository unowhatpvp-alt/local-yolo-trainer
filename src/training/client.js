export class TrainingClient{
 constructor({onProgress=()=>{},onCheckpoint=()=>{},onCrash=()=>{}}={}){this.onProgress=onProgress;this.onCheckpoint=onCheckpoint;this.onCrash=onCrash;this.worker=null;this.pending=new Map();this.sequence=0;}
 start(){
  if(this.worker)return;
  this.worker=new Worker(new URL('./worker.js',import.meta.url),{type:'module',name:'local-yolo-trainer'});
  this.worker.addEventListener('message',event=>{
   const message=event.data;
   if(message.type==='progress'){this.onProgress(message);return;}
   if(message.type==='checkpoint'){this.onCheckpoint(message);return;}
   const entry=this.pending.get(message.id);if(!entry)return;this.pending.delete(message.id);
   if(message.type==='error'){const error=new Error(message.error?.message||'Training worker failed.');error.code=message.error?.code;error.name=message.error?.name||'Error';entry.reject(error);}else entry.resolve(message.result);
  });
  this.worker.addEventListener('error',event=>this.crash(new Error(event.message||'The training process stopped unexpectedly.')));
  this.worker.addEventListener('messageerror',()=>this.crash(new Error('The training process returned an unreadable message.')));
 }
 call(type,payload={}){this.start();const id=++this.sequence;return new Promise((resolve,reject)=>{this.pending.set(id,{resolve,reject});try{this.worker.postMessage({id,type,payload});}catch(error){this.pending.delete(id);reject(error);}});}
 crash(error){for(const entry of this.pending.values())entry.reject(error);this.pending.clear();this.worker?.terminate();this.worker=null;this.onCrash(error);}
 async dispose(){if(!this.worker)return;try{await this.call('dispose');}finally{this.worker?.terminate();this.worker=null;}}
}
