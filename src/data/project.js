export const SCHEMA_VERSION=1;
export const DEFAULT_CONFIG={mode:'pretrained',resolution:128,batchSize:1,epochs:30,learningRate:0.003,augment:true,seed:12345,backend:'auto'};
export const uid=()=>crypto.randomUUID();
export function createProject({title='My project',className='My object'}={}){
 const now=new Date().toISOString();
 return {schemaVersion:1,id:uid(),title,mode:'single',classes:[className],images:[],config:{...DEFAULT_CONFIG},createdAt:now,updatedAt:now};
}
export function cleanName(value,fallback='Untitled'){
 const name=String(value??'').normalize('NFC').trim().replace(/[\u0000-\u001f\u007f]/g,'').slice(0,100);
 return name||fallback;
}
export function validateClasses(classes){
 if(!Array.isArray(classes)||classes.length<1||classes.length>32)throw new Error('CLASSES_RANGE');
 const seen=new Set();
 for(const name of classes){if(typeof name!=='string'||!name.trim()||name.length>80||/[\u0000-\u001f]/.test(name))throw new Error('CLASS_NAME');const key=name.normalize('NFC').trim().toLocaleLowerCase();if(seen.has(key))throw new Error('CLASS_DUPLICATE');seen.add(key);}
 return classes;
}
export function validateBoxes(boxes,width,height,classCount){
 if(!Array.isArray(boxes)||boxes.length>2000)throw new Error('BOXES_RANGE');
 const ids=new Set();
 for(const b of boxes){
  if(!b||typeof b.id!=='string'||!b.id||ids.has(b.id))throw new Error('BOX_ID');ids.add(b.id);
  if(!Number.isInteger(b.classId)||b.classId<0||b.classId>=classCount)throw new Error('BOX_CLASS');
  if(![b.x,b.y,b.width,b.height].every(Number.isFinite)||b.x<0||b.y<0||b.width<=0||b.height<=0||b.x+b.width>width+0.01||b.y+b.height>height+0.01)throw new Error('BOX_COORDINATES');
 }
 return boxes;
}
export function validateConfig(config){
 if(!config||!['pretrained','scratch','checkpoint'].includes(config.mode))throw new Error('TRAINING_MODE');
 if(config.resolution!==128)throw new Error('RESOLUTION_NOT_VERIFIED');
 if(![1,2,4,8,'auto'].includes(config.batchSize))throw new Error('BATCH_SIZE');
 if(!Number.isInteger(config.epochs)||config.epochs<1||config.epochs>1000)throw new Error('EPOCHS_RANGE');
 if(!Number.isFinite(config.learningRate)||config.learningRate<0.000001||config.learningRate>0.1)throw new Error('LEARNING_RATE');
 if(!Number.isInteger(config.seed)||config.seed<1||config.seed>0xffffffff||typeof config.augment!=='boolean')throw new Error('SEED_OR_AUGMENT');
 if(!['auto','cpu','wasm','webgl','webgpu'].includes(config.backend))throw new Error('BACKEND');
 return config;
}
export function isIsoTimestamp(value){
 return typeof value==='string'&&/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/.test(value)&&Number.isFinite(Date.parse(value))&&new Date(value).toISOString()===value;
}
export function validateProject(project){
 if(!project||project.schemaVersion!==1||typeof project.id!=='string'||!project.id||project.id.length>100)throw new Error('PROJECT_VERSION');
 if(typeof project.title!=='string'||project.title.length>100)throw new Error('PROJECT_TITLE');
 if(!isIsoTimestamp(project.createdAt)||!isIsoTimestamp(project.updatedAt))throw new Error('PROJECT_TIMESTAMPS');
 if(!['single','multi'].includes(project.mode))throw new Error('PROJECT_MODE');
 validateClasses(project.classes);validateConfig(project.config);
 if(project.mode==='single'&&project.classes.length!==1)throw new Error('SINGLE_CLASSES');
 if(!Array.isArray(project.images)||project.images.length>2000)throw new Error('IMAGES_RANGE');
 const ids=new Set();
 for(const image of project.images){
  if(!image||typeof image.id!=='string'||!image.id||ids.has(image.id))throw new Error('IMAGE_ID');ids.add(image.id);
  if(!Number.isInteger(image.width)||!Number.isInteger(image.height)||image.width<1||image.height<1||image.width*image.height>32_000_000)throw new Error('IMAGE_DIMENSIONS');
  if(typeof image.name!=='string'||image.name.length>250||!/^image\/(jpeg|png|webp)$/.test(image.mime)||!/^([a-f0-9]{64})$/.test(image.sha256))throw new Error('IMAGE_METADATA');
  if(typeof image.reviewed!=='boolean')throw new Error('IMAGE_REVIEW');
  if(image.split!==undefined&&!['train','validation'].includes(image.split))throw new Error('IMAGE_SPLIT');
  validateBoxes(image.boxes,image.width,image.height,project.classes.length);
 }
 return project;
}
export async function sha256(bytes){
 const view=bytes instanceof Blob?await bytes.arrayBuffer():bytes;
 const digest=await crypto.subtle.digest('SHA-256',view);
 return [...new Uint8Array(digest)].map(v=>v.toString(16).padStart(2,'0')).join('');
}
function shuffled(values,seed){
 let state=(seed>>>0)||0x9e3779b9;
 const rand=()=>{state^=state<<13;state^=state>>>17;state^=state<<5;return (state>>>0)/4294967296;};
 const out=[...values].sort();for(let i=out.length-1;i>0;i--){const j=Math.floor(rand()*(i+1));[out[i],out[j]]=[out[j],out[i]];}return out;
}
/** Identical images remain in one group. Validation never contains a training duplicate. */
export function assignSplits(project){
 const reviewed=project.images.filter(i=>i.reviewed),groups=[...new Set(reviewed.map(i=>i.sha256))];
 const order=shuffled(groups,project.config.seed),count=order.length<2?0:Math.max(1,Math.floor(order.length*0.2));
 const val=new Set(order.slice(0,count));
 for(const image of project.images)if(image.reviewed)image.split=val.has(image.sha256)?'validation':'train';else delete image.split;
 return {train:reviewed.filter(i=>i.split==='train').length,validation:reviewed.filter(i=>i.split==='validation').length,unreviewed:project.images.length-reviewed.length,unique:groups.length};
}
export async function datasetDigest(project){
 const canonical={classes:project.classes,resolution:project.config.resolution,seed:project.config.seed,images:project.images.filter(i=>i.reviewed).map(i=>({sha256:i.sha256,width:i.width,height:i.height,split:i.split,boxes:i.boxes.map(b=>({classId:b.classId,x:b.x,y:b.y,width:b.width,height:b.height})).sort((a,b)=>a.classId-b.classId||a.x-b.x||a.y-b.y)})).sort((a,b)=>JSON.stringify(a).localeCompare(JSON.stringify(b)))};
 return sha256(new TextEncoder().encode(JSON.stringify(canonical)));
}
export function escapeHtml(value){return String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
export function bytesLabel(bytes){return bytes<1024*1024?`${(bytes/1024).toFixed(0)} KB`:`${(bytes/1024/1024).toFixed(1)} MB`;}
