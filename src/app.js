import './styles.css';
import {tr,language,setLanguage,applyLocale} from './ui/locale.js';
import {cycleTheme,theme} from './ui/theme.js';
import {createAnnotationEditor} from './ui/annotation-editor.js';
import {createProject,cleanName,validateProject,validateClasses,validateConfig,assignSplits,escapeHtml as esc,bytesLabel} from './data/project.js';
import * as storage from './data/storage.js';
import {importImage,makeThumbnail,downloadBlob} from './data/media.js';
import {packProject,unpackProject,packCheckpoint,unpackCheckpoint,exportYoloDataset,validateCheckpoint} from './data/formats.js';
import {TrainingClient} from './training/client.js';
import {isAdditionalRun,trainingPlan} from './data/training-plan.js';

const app=document.querySelector('#app'),nav=document.querySelector('#step-nav'),$=selector=>document.querySelector(selector);
let project,step='images',imageIndex=0,activeClass=0,editor=null,renderToken=0,visibleImages=60,busy=false,uploadStatus=null,saveState='pending',storageError=null,lastError=null;
let checkpoint=null,checkpointPersist=Promise.resolve(),checkpointStorageError=null,training={running:false,status:'idle',progress:null,result:null},probe=null,testImage=null,testResult=null;
const mediaPending=new Map(),blobCache=new Map(),thumbnailUrls=new Set();
const nextFrame=()=>new Promise(resolve=>requestAnimationFrame(resolve));
const api=new TrainingClient({onProgress:message=>{training.progress=message;updateProgress();},onCheckpoint:message=>{
 checkpoint=message.checkpoint;
 const projectId=training.projectId||project.id;
 checkpointPersist=checkpointPersist.catch(()=>{}).then(()=>storage.saveCheckpoint(projectId,message.checkpoint)).then(()=>{checkpointStorageError=null;updateSaveState();}).catch(error=>{checkpointStorageError=error;updateSaveState();});
},onCrash:error=>{training.running=false;training.status='error';lastError=error;render();}});
const saves=new storage.SaveQueue(async value=>{
 for(const [id,entry] of [...mediaPending]){await storage.saveImage(id,entry.blob,entry.thumbnail);mediaPending.delete(id);}
 await storage.saveProject(value);
},(state,error)=>{saveState=state;if(error)storageError=error;else if(state==='saved')storageError=null;updateSaveState();});
const errors={
 IMAGE_FILE_TOO_LARGE:()=>tr('图片文件为空或超过 32 MB。请先缩小图片。','The picture is empty or larger than 32 MB. Resize it first.'),
 IMAGE_PIXELS_TOO_LARGE:()=>tr('图片超过 3200 万像素，请先缩小后再添加。','The picture exceeds 32 megapixels. Resize it before adding.'),
 IMAGE_DECODE_FAILED:()=>tr('无法读取这张图片，请尝试 JPEG、PNG 或 WebP。','Cannot read this picture. Try JPEG, PNG, or WebP.'),
 CLASS_DUPLICATE:()=>tr('这个物体名称已经存在。','That object name already exists.'),
 NO_REVIEWED_IMAGES:()=>tr('请先框选至少一张图片，或确认图片中没有目标。','Mark at least one picture, or confirm it contains no objects.'),
 CHECKPOINT_ARCHITECTURE:()=>tr('这不是兼容的训练检查点。用于识别的 ONNX 文件不能直接继续训练。','This is not a compatible training checkpoint. An inference ONNX cannot directly resume training.'),
 CHECKPOINT_INTEGRITY:()=>tr('模型文件不完整或已损坏，未替换当前模型。','The model file is incomplete or damaged. The current model was kept.'),
 PROJECT_IMAGE_INTEGRITY:()=>tr('项目图片校验失败，未覆盖现有项目。','Project image verification failed. Existing projects were kept.'),
 ARCHIVE_UNSAFE:()=>tr('备份包含不安全的路径或超出大小限制，已拒绝导入。','The backup contains unsafe paths or exceeds size limits and was rejected.'),
 MODEL_REQUIRED:()=>tr('请先训练模型，或导入兼容的训练检查点。','Train a model or import a compatible training checkpoint first.'),
 INVALID_CHECKPOINT:()=>tr('检查点结构、训练状态或模型版本不兼容，当前模型已保留。','The checkpoint structure, training state, or model version is incompatible. The current model was kept.'),
 CHECKPOINT_CLASS_MISMATCH:()=>tr('模型类别与当前标注不一致。请选择新的训练方式，或在新项目中导入；已有标注已保留。','Model classes differ from these annotations. Start a new training mode or import into a new project; annotations were kept.'),
 UNSUPPORTED_BACKEND:()=>tr('当前计算方式无法可靠训练这种模型，请选择“自动”或 CPU 后重试。','This backend cannot reliably train this model. Choose Auto or CPU and retry.'),
 NO_TRAIN_BOXES:()=>tr('用于学习的图片中至少需要一个已框选目标。','Mark at least one object in a training picture.'),
 INVALID_IMAGE:()=>tr('无法读取或处理这张图片，请尝试另一张 JPEG、PNG 或 WebP。','This picture could not be read or processed. Try another JPEG, PNG, or WebP.'),
 IMAGE_DIMENSION_MISMATCH:()=>tr('图片方向或尺寸与标注不一致，请重新导入；已有数据已保留。','Picture orientation or dimensions differ from the annotations. Reimport it; existing data was kept.'),
 BUSY:()=>tr('正在处理前一个操作，请稍候或先暂停学习。','A previous operation is still running. Wait or pause learning first.'),
 TARGET_COLLISION:()=>tr('同一小区域中的目标过多，这个轻量模型无法表达全部标注。请查看错误详情。','Too many targets share one small region for this lightweight model. See details.'),
 DATA_SPLIT_LEAKAGE:()=>tr('学习和验证图片中存在相同内容。请移除重复项后重试，原标注已保留。','Training and validation contain the same picture content. Remove duplicates and retry; annotations were kept.'),
 EPOCHS_RANGE:()=>tr('总目标轮数须在 1 至 1000 之间；继续学习会加上已完成的轮数。','The total target must be 1–1000 epochs. Continuing adds epochs to the completed run.'),
 DATASET_MEMORY_LIMIT:()=>tr('这个项目超过当前训练内存预算，请减少本次使用的图片。标注仍会保留。','This project exceeds the training memory budget. Use fewer pictures for this run; annotations are kept.'),
};
function message(error){const code=error?.code||error?.message;return errors[code]?.()||tr('操作未完成，原有数据和最近的有效模型已保留。','The operation did not finish. Existing data and the last valid model were kept.');}
function toast(text){const node=$('#toast');node.textContent=text;node.hidden=false;clearTimeout(toast.timer);toast.timer=setTimeout(()=>node.hidden=true,6000);}
function handle(error){lastError=error;toast(message(error));renderError();}
const safe=fn=>event=>Promise.resolve().then(()=>fn(event)).catch(handle);
function bind(selector,event,fn,root=app){const node=root.querySelector(selector);if(node)node.addEventListener(event,safe(fn));}
function stamp(){project.updatedAt=new Date().toISOString();saves.enqueue(project);updateNav();}
function updateSaveState(){const failed=storageError||checkpointStorageError;const node=$('#save-status');if(node)node.textContent=failed||saveState==='error'?tr('未能保存，请导出备份','Not saved — export a backup'):saveState==='saved'?tr('已保存在此浏览器','Saved in this browser'):tr('正在保存…','Saving…');const warning=$('#storage-warning');if(warning){warning.hidden=!failed;warning.textContent=tr('本地保存尚未完成。请保持页面打开并导出项目备份；浏览器本地存储不是永久备份。','Local saving has not completed. Keep this page open and export a project backup; browser storage is not a permanent backup.');}}
function updateNav(){for(const button of nav.querySelectorAll('[data-step]')){button.setAttribute('aria-current',button.dataset.step===step?'step':'false');button.disabled=busy||training.running&&button.dataset.step!=='train';}$('#project-menu').disabled=busy||training.running;app.inert=busy;}
function renderError(){const node=$('#operation-error');if(!node)return;node.hidden=!lastError;node.innerHTML=lastError?`<p>${esc(message(lastError))}</p><details><summary>${tr('详细原因','Details')}</summary><p>${esc(lastError.message)}</p></details><button class="button quiet" id="dismiss-error">${tr('关闭提示','Dismiss')}</button>`:'';node.querySelector('#dismiss-error')?.addEventListener('click',()=>{lastError=null;renderError();});}
function frame(title,description,content){return `<section class="page"><div class="page-header"><div><p class="eyebrow">${esc(project.title)}</p><h1>${title}</h1>${description?`<p class="muted">${description}</p>`:''}</div><span id="save-status" class="muted" role="status"></span></div><p id="storage-warning" class="notice" role="status" hidden></p><div id="operation-error" class="notice error" role="alert" hidden></div>${content}</section>`;}
function releaseView(){editor?.destroy();editor=null;for(const url of thumbnailUrls)URL.revokeObjectURL(url);thumbnailUrls.clear();}
async function render(){
 if(!project)return;
 const token=++renderToken;releaseView();
 if(step==='images')renderImages();else if(step==='annotate')await renderAnnotate(token);else if(step==='train')renderTrain();else await renderTest(token);
 updateNav();updateSaveState();renderError();applyLocale();
}
async function getBlob(id){if(mediaPending.has(id))return mediaPending.get(id).blob;if(blobCache.has(id))return blobCache.get(id);const blob=await storage.getImage(id);if(!(blob instanceof Blob))throw new Error('PROJECT_IMAGE_INTEGRITY');blobCache.set(id,blob);if(blobCache.size>5)blobCache.delete(blobCache.keys().next().value);return blob;}
async function getThumb(id){return mediaPending.get(id)?.thumbnail||await storage.getThumbnail(id)||await makeThumbnail(await getBlob(id));}
function chooseFile({accept,multiple=false,onFiles}){const input=document.createElement('input');input.type='file';input.accept=accept;input.multiple=multiple;input.addEventListener('change',safe(()=>onFiles([...input.files])));input.click();}
function renderImages(){
 const rows=project.images.slice(0,visibleImages);
 app.innerHTML=frame(tr('教它认识你的物体','Teach it to recognise your objects'),tr('先添加图片，再框出你希望识别的物体。图片与训练都留在这台设备。','Add pictures, then mark the objects you want to recognise. Pictures and training stay on this device.'),`
 <div class="row spread"><label class="field project-title"><span>${tr('项目名称','Project name')}</span><input id="project-title" value="${esc(project.title)}" maxlength="100"></label><span>${project.images.length} ${tr('张图片','pictures')}</span></div>
 <div id="drop-zone" class="drop-zone" tabindex="0" role="button" aria-label="${tr('添加图片，也可拖放图片','Add pictures, or drop them here')}"><span class="drop-symbol" aria-hidden="true">＋</span><strong>${tr('添加图片','Add pictures')}</strong><span class="muted">${tr('点击选择、拖放，或从相册添加','Choose files, drop them here, or use your photo library')}</span><small>JPEG · PNG · WebP · ${tr('其他格式取决于浏览器','Other formats depend on your browser')}</small></div>
 ${uploadStatus?`<p role="status">${esc(uploadStatus)}</p>`:''}
 <div class="row spread"><span class="muted">${tr('图片会按正确方向处理，不修改原始文件。每张最多 32 MB / 3200 万像素。','Orientation is corrected without modifying original files. Maximum 32 MB / 32 megapixels each.')}</span>${project.images.length?`<button id="go-annotate" class="button primary">${tr('开始框选','Start marking')} →</button>`:''}</div>
 ${rows.length?`<div class="image-grid">${rows.map((image,index)=>`<article class="image-card"><button class="image-open" data-image="${index}" aria-label="${esc(tr('标注 ','Mark ')+image.name)}"><img data-thumb="${image.id}" alt="${esc(image.name)}" loading="lazy"><span class="image-caption"><span>${esc(image.name)}</span><small>${image.reviewed?`${image.boxes.length} ${tr('个框','boxes')}`:tr('待框选','Not marked')}</small></span></button><button class="button quiet image-delete" data-delete="${image.id}" aria-label="${esc(tr('删除 ','Delete ')+image.name)}">×</button></article>`).join('')}</div>`:`<div class="empty-state"><p>${tr('同一物体的不同角度、光照和背景，能帮助模型学得更好。','Different angles, lighting, and backgrounds help your model learn.')}</p><p>${tr('一张图片也可以体验流程；可靠识别需要更多、多样的图片。','One picture can demonstrate the process; reliable recognition needs more varied pictures.')}</p></div>`}
 ${project.images.length>visibleImages?`<button id="more-images" class="button">${tr('继续显示图片','Show more pictures')}</button>`:''}
 <p class="muted backup-hint">${tr('浏览器可能清理本地数据。通过右上角“项目”导出完整备份。','Browsers may clear local data. Export a full backup from Project in the top corner.')}</p>`);
 bind('#project-title','change',event=>{project.title=cleanName(event.target.value,tr('我的项目','My project'));stamp();event.target.value=project.title;});
 const add=()=>{if(!busy)chooseFile({accept:'image/*',multiple:true,onFiles:uploadImages});};
 bind('#drop-zone','click',add);bind('#drop-zone','keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();add();}});
 bind('#drop-zone','dragover',event=>{event.preventDefault();event.currentTarget.classList.add('drag-over');});bind('#drop-zone','dragleave',event=>event.currentTarget.classList.remove('drag-over'));
 bind('#drop-zone','drop',event=>{event.preventDefault();if(!busy)return uploadImages([...event.dataTransfer.files]);});
 bind('#go-annotate','click',()=>navigate('annotate'));bind('#more-images','click',()=>{visibleImages+=60;render();});
 for(const button of app.querySelectorAll('[data-image]'))button.addEventListener('click',()=>{imageIndex=Number(button.dataset.image);navigate('annotate');});
 for(const button of app.querySelectorAll('[data-delete]'))button.addEventListener('click',safe(()=>removeImage(button.dataset.delete)));
 const token=renderToken;
 for(const image of rows)getThumb(image.id).then(blob=>{if(renderToken!==token)return;const node=app.querySelector(`[data-thumb="${CSS.escape(image.id)}"]`);if(node){const url=URL.createObjectURL(blob);thumbnailUrls.add(url);node.src=url;}}).catch(handle);
 if(busy)for(const node of app.querySelectorAll('button,input,#drop-zone')){node.disabled=true;node.setAttribute('aria-disabled','true');}
}
async function uploadImages(files){
 if(training.running||busy)return;busy=true;lastError=null;let added=0,duplicates=0,failed=0;const failures=[];
 try{
  for(let i=0;i<files.length;i++){
   uploadStatus=tr(`正在处理 ${i+1} / ${files.length}…`,`Processing ${i+1} / ${files.length}…`);await render();await nextFrame();
   try{
    if(project.images.length>=2000)throw new Error('IMAGES_RANGE');
    const entry=await importImage(files[i]);
    if(project.images.some(image=>image.sha256===entry.image.sha256||image.sourceSha256===entry.image.sourceSha256)){duplicates++;continue;}
    mediaPending.set(entry.image.id,entry);project.images.push(entry.image);added++;stamp();
   }catch(error){failed++;failures.push({name:files[i].name,message:message(error)});}
  }
  try{await saves.flush();}catch{/* Keep all pending images and make backup available. */}
 }finally{busy=false;uploadStatus=null;await render();}
 toast(tr(`已添加 ${added} 张，重复 ${duplicates} 张，失败 ${failed} 张。`,`Added ${added}; duplicates ${duplicates}; failed ${failed}.`));
 if(failures.length)await dialog(tr('部分图片未添加','Some pictures were not added'),`<ul>${failures.slice(0,20).map(f=>`<li>${esc(f.name)} — ${esc(f.message)}</li>`).join('')}</ul>`,[{label:tr('知道了','OK'),value:true}]);
}
async function removeImage(id){
 if(busy||training.running)return;
 const image=project.images.find(i=>i.id===id);if(!image)return;
 const yes=await confirmDialog(tr('移除这张图片？','Remove this picture?'),tr(`“${image.name}”及它的${image.boxes.length}个标注将从当前项目移除。磁盘上的原始文件不会改变。`,`“${image.name}” and its ${image.boxes.length} annotations will be removed from this project. The original file on disk is unchanged.`));
 if(!yes)return;busy=true;updateNav();try{
  await saves.flush();
  const next=structuredClone(project);next.images=next.images.filter(i=>i.id!==id);next.updatedAt=new Date().toISOString();
  project=await storage.removeImageConfirmed(next,id);blobCache.delete(id);mediaPending.delete(id);
  imageIndex=Math.min(imageIndex,Math.max(0,project.images.length-1));saveState='saved';
 }finally{busy=false;await render();}
}
function classControls(){return `<div class="row spread"><label class="field"><span>${tr('要识别的物体','Objects to recognise')}</span><select id="class-mode"><option value="single" ${project.mode==='single'?'selected':''}>${tr('一种物体','One kind of object')}</option><option value="multi" ${project.mode==='multi'?'selected':''}>${tr('多种物体','Several kinds of objects')}</option></select></label><div class="row"><label class="field"><span>${tr('当前物体','Current object')}</span><select id="active-class">${project.classes.map((name,i)=>`<option value="${i}" ${i===activeClass?'selected':''}>${esc(name)}</option>`).join('')}</select></label><button class="button" id="edit-classes">${tr('编辑名称','Edit names')}</button></div></div>`;}
async function renderAnnotate(token){
 if(!project.images.length){step='images';renderImages();return;}
 imageIndex=Math.max(0,Math.min(imageIndex,project.images.length-1));const image=project.images[imageIndex];
 app.innerHTML=frame(tr('框出你希望它认识的物体','Mark the objects you want it to recognise'),tr('每个物体画一个框。下一张会自动沿用物体名称。','Draw one box around each object. The same names are reused on the next picture.'),`${classControls()}<div class="row spread"><button id="previous-image" class="button" ${imageIndex===0?'disabled':''}>← ${tr('上一张','Previous')}</button><span>${imageIndex+1} / ${project.images.length} · ${esc(image.name)}</span><button id="next-image" class="button" ${imageIndex===project.images.length-1?'disabled':''}>${tr('下一张','Next')} →</button></div><div id="annotation-editor"></div><div class="row spread"><label class="checkbox-row"><input type="checkbox" id="reviewed" ${image.reviewed?'checked':''}>${tr('将这张图用于学习（无目标时也可勾选）','Use this picture for training, including if it has no objects')}</label><span id="annotation-count">${image.boxes.length} ${tr('个框','boxes')}</span></div><div class="row spread"><span class="muted">${project.images.filter(i=>i.reviewed).length} / ${project.images.length} ${tr('张用于学习；标注会自动保存。','pictures selected for training; annotations save automatically.')}</span><button class="button primary" id="go-train">${tr('准备学习','Prepare to learn')} →</button></div>`);
 bind('#previous-image','click',()=>{imageIndex--;render();});bind('#next-image','click',()=>{imageIndex++;render();});bind('#go-train','click',()=>navigate('train'));
 bind('#reviewed','change',event=>{image.reviewed=event.target.checked;stamp();});
 bind('#active-class','change',event=>{activeClass=Number(event.target.value);editor?.setActiveClass(activeClass);});
 bind('#class-mode','change',event=>changeMode(event.target.value));bind('#edit-classes','click',editClasses);
 const controls=['#active-class','#class-mode','#edit-classes'].map(selector=>$(selector));
 for(const control of controls)control.disabled=true;
 const blob=await getBlob(image.id);if(token!==renderToken)return;
 const view=createAnnotationEditor({root:$('#annotation-editor'),onChange:boxes=>{image.boxes=boxes;image.reviewed=true;const checked=$('#reviewed');if(checked)checked.checked=true;const count=$('#annotation-count');if(count)count.textContent=`${boxes.length} ${tr('个框','boxes')}`;stamp();}});editor=view;
 await view.load({blob,width:image.width,height:image.height,boxes:image.boxes,classes:project.classes,activeClassIndex:activeClass});
 if(token!==renderToken){view.destroy();return;}
 for(const control of controls)control.disabled=false;
}
async function changeMode(mode){
 if(mode==='multi'){project.mode='multi';stamp();render();return;}
 const used=[...new Set(project.images.flatMap(i=>i.boxes.map(b=>b.classId)))];
 if(used.length>1){toast(tr('已有多个类别的标注，请先将方框调整为同一类别。所有标注已保留。','Several categories are already marked. Change boxes to one category first. All annotations were kept.'));render();return;}
 const keep=used[0]??activeClass;
 if(project.classes.length>1&&!await confirmDialog(tr('切换为一种物体','Use one kind of object'),tr('将保留正在使用的物体名称和全部方框，移除未使用的类别名称。','Keep the used object name and every box, and remove unused category names.'))){render();return;}
 project.classes=[project.classes[keep]];for(const image of project.images)for(const box of image.boxes)box.classId=0;activeClass=0;project.mode='single';stamp();render();
}
async function editClasses(){
 const content=`<div class="stack">${project.classes.map((name,i)=>`<label class="field"><span>${tr('物体','Object')} ${i+1}</span><input data-class-name="${i}" value="${esc(name)}" maxlength="80"></label>`).join('')}${project.mode==='multi'&&project.classes.length<32?`<label class="field"><span>${tr('添加新物体（可留空）','Add another object (optional)')}</span><input id="new-class" maxlength="80" placeholder="${tr('例如：苹果','For example: apple')}"></label>`:''}<p class="muted">${tr('名称会应用到已有方框。已有标注不会因为重命名被删除。','Names apply to existing boxes. Renaming does not delete annotations.')}</p></div>`;
 const result=await dialog(tr('物体名称','Object names'),content,[{label:tr('取消','Cancel'),value:null},{label:tr('保存','Save'),value:true,primary:true}],node=>{const names=[...node.querySelectorAll('[data-class-name]')].map(input=>input.value.trim());const extra=node.querySelector('#new-class')?.value.trim();if(extra)names.push(extra);validateClasses(names);return names;});
 if(result){project.classes=result;stamp();render();}
}
function renderTrain(){
 const counts=assignSplits(structuredClone(project)),hasCheckpoint=Boolean(checkpoint),running=training.running;
 app.innerHTML=frame(tr('让模型学习你的图片','Let the model learn your pictures'),tr('训练使用本设备计算。你可以先使用推荐设置。','Training uses this device. Start with the recommended settings.'),`
 <div class="metric-grid"><div><strong>${counts.train}</strong><span>${tr('张用于学习','training pictures')}</span></div><div><strong>${counts.validation}</strong><span>${tr('张用于验证','validation pictures')}</span></div><div><strong>${project.classes.length}</strong><span>${tr('种物体','object categories')}</span></div></div>
 ${counts.unreviewed?`<p class="notice">${tr(`${counts.unreviewed} 张尚未选择用于学习，本次不会参与。`,`${counts.unreviewed} pictures are not selected for training and will not be used in this run.`)}</p>`:''}
 ${counts.validation===0?`<p class="notice">${tr('当前没有独立验证图片，可以体验流程，但无法评估对新场景的识别能力。','There are no held-out validation pictures. You can try the process, but performance on new scenes cannot be measured.')}</p>`:''}
 <div class="fields"><label class="field"><span>${tr('如何开始','How to start')}</span><select id="training-mode" ${running?'disabled':''}><option value="pretrained" ${project.config.mode==='pretrained'?'selected':''}>${tr('预训练特征 + 轻量检测头（推荐）','Pretrained features + lightweight detector (recommended)')}</option><option value="checkpoint" ${project.config.mode==='checkpoint'?'selected':''} ${!hasCheckpoint?'disabled':''}>${tr('继续兼容的训练检查点','Continue a compatible checkpoint')}</option><option value="scratch" ${project.config.mode==='scratch'?'selected':''}>${tr('轻量检测器，从零开始','Small detector, from scratch')}</option></select></label><button id="import-model" class="button" ${running?'disabled':''}>${tr('导入训练检查点','Import training checkpoint')}</button></div>
 <p class="muted">${tr('推荐方式冻结 YOLO26n 的空间特征，只训练新的检测头。从零方式训练独立的小型卷积检测器。完整 YOLO26 训练尚不支持。','The recommended mode freezes YOLO26n spatial features and trains a new detection head. From scratch trains a separate small convolutional detector. Full YOLO26 training is unsupported.')}</p>
 ${project.config.mode==='scratch'?`<p class="notice">${tr('从零学习通常需要更多图片和更长时间；很小的数据集不保证效果。','Learning from scratch usually needs more pictures and more time. Tiny datasets do not guarantee useful results.')}</p>`:''}
 <details class="panel" id="advanced-training"><summary>${tr('高级设置','Advanced settings')}</summary><div class="fields"><label class="field"><span>${tr('学习轮数','Epochs')}</span><input id="epochs" type="number" min="1" max="1000" value="${project.config.epochs}" ${running?'disabled':''}></label><label class="field"><span>Batch size</span><select id="batch-size" ${running?'disabled':''}>${[1,2,4,8,'auto'].map(value=>`<option value="${value}" ${project.config.batchSize===value?'selected':''}>${value==='auto'?tr('自动（从 1 开始）','Auto (starts at 1)'):value}</option>`).join('')}</select></label><label class="field"><span>${tr('学习率','Learning rate')}</span><input id="learning-rate" type="number" min="0.000001" max="0.1" step="0.0001" value="${project.config.learningRate}" ${running?'disabled':''}></label><label class="field"><span>${tr('计算方式','Compute backend')}</span><select id="training-backend" ${running?'disabled':''}>${['auto','webgpu','wasm','webgl','cpu'].map(value=>{const tested=probe?.backends?.find(b=>b.backend===value);return `<option value="${value}" ${project.config.backend===value?'selected':''} ${tested&&!tested.supported?'disabled':''}>${value==='auto'?tr('自动选择已通过探测的方式','Auto: verified backend'):value.toUpperCase()}${tested&&!tested.supported?tr('（此模式不可用）',' (unavailable for this mode)'):''}</option>`;}).join('')}</select></label><label class="field"><span>${tr('输入尺寸','Input size')}</span><select disabled><option>128 × 128</option><option disabled>192 × 192 — ${tr('未验证','not verified')}</option></select></label><label class="checkbox-row"><input type="checkbox" id="augmentation" ${project.config.augment?'checked':''} ${running?'disabled':''}>${tr('随机水平翻转图片与方框','Randomly flip pictures and boxes horizontally')}</label></div><p class="muted">${tr('批次更大不一定更快。自动模式保守使用 1。此设备可用性由真实梯度探测确定，不猜测显存。','Larger batches are not necessarily faster. Auto conservatively uses 1. Availability is based on actual gradient checks, not guessed GPU memory.')}</p><button id="probe-backends" class="button" ${running?'disabled':''}>${tr('检查本机计算支持','Check this device')}</button><div id="backend-report">${probeReport()}</div></details>
 <section id="training-progress" class="training-progress" ${!training.progress?'hidden':''}><h2 id="training-state"></h2><progress id="training-bar" max="1" value="0"></progress><div id="training-numbers" class="muted"></div><p id="training-eta" class="muted"></p><details><summary>${tr('实际计算指标','Actual computation metrics')}</summary><pre id="training-metrics"></pre></details></section>
 <div class="row spread"><span class="muted">${tr('离开页面时会请求暂停。关闭浏览器后训练不会继续。','Leaving the page requests a pause. Training does not continue after closing the browser.')}</span>${running?`<div class="row"><button id="pause-training" class="button">${tr('暂停并保存','Pause & save')}</button><button id="stop-training" class="button">${tr('停止并保存','Stop & save')}</button></div>`:`<button id="start-training" class="button primary" ${counts.train===0?'disabled':''}>${project.config.mode==='checkpoint'?tr('继续学习','Continue learning'):tr('开始学习','Start learning')} →</button>`}</div>
 ${checkpoint&&!running?`<div class="row spread"><span>${tr('已有可保存的训练模型。','A trained checkpoint is available.')}</span><button class="button" id="go-test">${tr('测试与保存','Test & save')} →</button></div>`:''}`);
 $('#epochs').previousElementSibling.textContent=isAdditionalRun(project.config,checkpoint)?tr('再学习多少轮','Additional epochs'):project.config.mode==='checkpoint'?tr('总目标轮数','Total target epochs'):tr('学习轮数','Epochs');
 const change=()=>{const next={...project.config,epochs:Number($('#epochs').value),batchSize:$('#batch-size').value==='auto'?'auto':Number($('#batch-size').value),learningRate:Number($('#learning-rate').value),backend:$('#training-backend').value,augment:$('#augmentation').checked};validateConfig(next);trainingPlan(next,checkpoint);project.config=next;stamp();};
 for(const selector of ['#epochs','#batch-size','#learning-rate','#training-backend','#augmentation'])bind(selector,'change',change);
 bind('#training-mode','change',event=>{project.config.mode=event.target.value;probe=null;stamp();render();});
 bind('#import-model','click',importModel);bind('#probe-backends','click',checkBackends);bind('#start-training','click',startTraining);bind('#go-test','click',()=>navigate('test'));
 bind('#pause-training','click',()=>api.call('pause'));bind('#stop-training','click',()=>api.call('stop'));updateProgress();
 if(busy)for(const control of app.querySelectorAll('button,input,select'))control.disabled=true;
}
function probeReport(){if(!probe)return '';return `<ul>${probe.backends.map(item=>`<li>${esc(item.backend.toUpperCase())} · ${item.supported?tr('真实梯度探测通过','Actual gradient check passed'):tr('不可用','Unavailable')}${!item.supported?`<details><summary>${tr('原因','Reason')}</summary>${esc(item.reason)}</details>`:''}</li>`).join('')}</ul>`;}
async function checkBackends(){if(busy||training.running)return;busy=true;await render();$('#advanced-training').open=true;const node=$('#probe-backends');node.disabled=true;node.textContent=tr('正在执行真实计算…','Running real computation…');try{probe=await api.call('probe',{mode:project.config.mode==='checkpoint'?(checkpoint?.architecture==='tiny-grid-v1'?'scratch':'pretrained'):project.config.mode});}finally{busy=false;await render();if(step==='train')$('#advanced-training').open=true;}}
function updateProgress(){
 const p=training.progress;if(!p||step!=='train')return;const section=$('#training-progress');if(!section)return;section.hidden=false;
 const preparing=p.state==='preparing',fraction=Math.max(0,Math.min(1,p.progress||0));
 $('#training-state').textContent=preparing?tr(`正在读取图片 ${p.completedImages} / ${p.totalImages}`,`Reading pictures ${p.completedImages} / ${p.totalImages}`):p.state==='training'?tr(`正在学习你的图片：${Math.floor(fraction*100)}%`,`Learning your pictures: ${Math.floor(fraction*100)}%`):p.state==='completed'?tr('这次学习已完成','This learning run finished'):p.state==='paused'?tr('已暂停，模型可保存','Paused; your model can be saved'):tr('已停止，模型可保存','Stopped; your model can be saved');
 $('#training-bar').value=fraction;$('#training-numbers').textContent=preparing?'':tr(`第 ${p.epoch??0} / ${p.totalEpochs??project.config.epochs} 轮 · ${p.step??0} 次更新 · ${(p.backend||'').toUpperCase()}`,`Epoch ${p.epoch??0} / ${p.totalEpochs??project.config.epochs} · ${p.step??0} updates · ${(p.backend||'').toUpperCase()}`);
 $('#training-eta').textContent=p.state==='training'?(p.remainingMs==null?tr('剩余时间：正在估算','Time remaining: estimating'):tr(`剩余约 ${Math.ceil(p.remainingMs/1000)} 秒 · 已用 ${Math.round((p.elapsedMs||0)/1000)} 秒`,`About ${Math.ceil(p.remainingMs/1000)} seconds remaining · ${Math.round((p.elapsedMs||0)/1000)} seconds elapsed`)):'';
 $('#training-metrics').textContent=JSON.stringify({loss:p.loss??null,gradientNorm:p.gradient?.norm??p.gradient?.gradientNorm??null,backend:p.backend??null,step:p.step??0},null,2);
}
async function startTraining(){
 if(busy||training.running)return;validateProject(project);const counts=assignSplits(project);if(!counts.train)throw new Error('NO_REVIEWED_IMAGES');
 if(project.config.mode==='checkpoint'&&!checkpoint)throw new Error('MODEL_REQUIRED');
 const request=trainingPlan(project.config,checkpoint);training={running:true,status:'training',projectId:project.id,progress:null,result:null};stamp();lastError=null;await render();
 try{
  await saves.flush().catch(()=>{});
  const samples=[];for(const image of project.images.filter(i=>i.reviewed))samples.push({...image,blob:await getBlob(image.id)});
  const result=await api.call('train',{samples,classes:project.classes,...request,checkpoint:project.config.mode==='checkpoint'?checkpoint:undefined});
  checkpoint=result.checkpoint;training.result=result;training.status=result.status;await checkpointPersist;project.config.mode='checkpoint';stamp();
  toast(result.status==='completed'?tr('学习已完成，上传新图片试试看。','Learning finished. Try a new picture.'):tr('已保留最近的训练模型。','The latest training model was kept.'));
 }catch(error){training.status='error';handle(error);}finally{training.running=false;await render();}
}
async function importModel(){
 chooseFile({accept:'.lymodel,.zip,.onnx',onFiles:async files=>{
  const file=files[0];if(!file||busy||training.running)return;busy=true;updateNav();try{if(/\.onnx$/i.test(file.name))throw new Error('CHECKPOINT_ARCHITECTURE');
  const next=validateCheckpoint(await unpackCheckpoint(file));
  if(next.classes.some((value,i)=>value!==project.classes[i])||next.classes.length!==project.classes.length){
   if(project.images.some(image=>image.boxes.length))throw new Error(tr('模型类别与现有标注不一致。请新建项目后导入，当前标注未变。','Model categories differ from existing annotations. Import into a new project; existing annotations were kept.'));
   project.classes=[...next.classes];project.mode=next.classes.length===1?'single':'multi';
  }
  // Validate the complete portable state before replacing the current model.
  checkpoint=next;project.config={...project.config,...next.training.config,mode:'checkpoint'};stamp();await storage.saveCheckpoint(project.id,next).then(()=>{checkpointStorageError=null;}).catch(error=>{checkpointStorageError=error;});toast(tr('训练检查点已读取，继续学习时会校验图片与标注。','Checkpoint loaded. Pictures and annotations are verified before resuming.'));await render();
  }finally{busy=false;updateNav();updateSaveState();}
 }});
}
async function renderTest(token){
 const result=training.result,metrics=result?.metrics??checkpoint?.training.history.at(-1)?.metrics;
 app.innerHTML=frame(tr('试试它学到了什么','See what it has learned'),tr('用未参与学习的新图片测试，效果更有参考价值。','Use a new picture that was not used for training.'),`
 ${!checkpoint?`<div class="empty-state"><p>${tr('先完成一次学习，或导入兼容训练检查点。','Train a model, or import a compatible checkpoint.')}</p><button id="back-train" class="button primary">${tr('前往学习','Go to training')}</button><button id="import-model" class="button">${tr('导入训练检查点','Import checkpoint')}</button></div>`:`<div class="row spread"><button id="test-picture" class="button primary">${tr('用新图片测试','Test a new picture')}</button><div class="row"><button id="save-model" class="button">${tr('保存可继续学习的模型','Save trainable model')}</button><button id="export-onnx" class="button">${tr('导出推理模型 ONNX','Export inference ONNX')}</button></div></div><p class="muted">${tr('训练检查点可继续学习；ONNX 用于识别，可导入现有 Local YOLO 推理工具。','A checkpoint can resume learning. ONNX is for inference and can be imported into the existing Local YOLO tool.')}</p><div id="test-preview"></div><p id="test-summary" role="status">${testResult?esc(tr(`${testResult.items.length} 个结果 · ${Math.round(testResult.latency)} ms · ${testResult.backend}`,`${testResult.items.length} results · ${Math.round(testResult.latency)} ms · ${testResult.backend}`)):tr('选择一张图片查看实际检测结果。','Choose a picture to see actual detections.')}</p><details class="panel"><summary>${tr('本次学习的实际记录','Actual training record')}</summary><pre>${esc(JSON.stringify({architecture:checkpoint.architecture,step:checkpoint.training.step,validation:metrics??null,evidence:result?.evidence??null},null,2))}</pre></details><p class="notice">${metrics?.images?tr('验证结果只代表这些图片；增加不同背景、角度与光照的图片后再测试。','Validation results describe these pictures only. Add varied backgrounds, angles, and lighting, then test again.'):tr('没有独立验证结果。可以测试和保存模型，但不要据此认为已具备可靠泛化能力。','No independent validation result is available. You can test and save the model, but reliable generalisation has not been established.')}</p><div class="row"><button id="continue-learning" class="button">${tr('继续学习','Continue learning')}</button><a class="button quiet" href="https://unowhatpvp-alt.github.io/local-yolo/" target="_blank" rel="noopener">${tr('打开推理工具','Open inference tool')} ↗</a></div>`}`);
 bind('#back-train','click',()=>navigate('train'));bind('#import-model','click',importModel);bind('#test-picture','click',()=>chooseFile({accept:'image/*',onFiles:files=>testPicture(files[0])}));
 bind('#save-model','click',saveModel);bind('#export-onnx','click',exportOnnx);bind('#continue-learning','click',()=>{project.config.mode='checkpoint';stamp();navigate('train');});
 if(testImage&&checkpoint){const node=$('#test-preview');if(!node)return;const view=createAnnotationEditor({root:node,readonly:true,onChange:()=>{}});editor=view;await view.load({blob:testImage.blob,width:testImage.width,height:testImage.height,boxes:(testResult?.items||[]).map((item,i)=>({id:`prediction-${i}`,classId:item.classId,...item.box,score:item.score})),classes:checkpoint.classes,activeClassIndex:0});if(token!==renderToken)view.destroy();}
}
async function testPicture(file){if(!file||busy||training.running)return;busy=true;updateNav();try{const entry=await importImage(file);const result=await api.call('predict',{blob:entry.blob,checkpoint,confidence:0.25});testImage={blob:entry.blob,width:entry.image.width,height:entry.image.height};testResult=result;lastError=null;await render();}finally{busy=false;updateNav();}}
async function saveModel(){if(!checkpoint)throw new Error('MODEL_REQUIRED');const blob=await packCheckpoint(checkpoint);downloadBlob(blob,'local-yolo-model.lymodel');toast(tr('已生成可重新加载的训练检查点。','A reloadable training checkpoint was generated.'));}
async function exportOnnx(){if(!checkpoint)throw new Error('MODEL_REQUIRED');const result=await api.call('export',{checkpoint});downloadBlob(new Blob([result.bytes],{type:result.mime}),result.filename);toast(tr('已生成有效 ONNX 模型；请在推理工具中选择“导入本地 ONNX”。','An ONNX model was generated. Choose Import local ONNX in the inference tool.'));}
async function navigate(next){if(busy||training.running&&next!=='train')return;if(next==='annotate'&&!project.images.length)next='images';step=next;location.hash=next;await render();}
async function dialog(title,content,actions,readValue){
 return new Promise(resolve=>{
  const node=document.createElement('dialog');node.className='dialog';node.innerHTML=`<form method="dialog"><h2>${esc(title)}</h2><div class="dialog-content">${content}</div><p class="dialog-error notice" role="alert" hidden></p><div class="row dialog-actions">${actions.map((action,i)=>`<button type="button" class="button ${action.primary?'primary':''}" data-action="${i}">${esc(action.label)}</button>`).join('')}</div></form>`;
  document.body.append(node);let completed=false;
  const finish=value=>{if(completed)return;completed=true;node.close();node.remove();resolve(value);};
  node.addEventListener('cancel',event=>{event.preventDefault();finish(null);});
  node.addEventListener('click',event=>{if(event.target===node){const rect=node.getBoundingClientRect();if(event.clientX<rect.left||event.clientX>rect.right||event.clientY<rect.top||event.clientY>rect.bottom)finish(null);}});
  for(const button of node.querySelectorAll('[data-action]'))button.addEventListener('click',()=>{try{const action=actions[Number(button.dataset.action)];finish(action.value&&readValue?readValue(node):action.value);}catch(error){const field=node.querySelector('.dialog-error');field.hidden=false;field.textContent=message(error);}});
  node.showModal();
 });
}
const confirmDialog=(title,text)=>dialog(title,`<p>${esc(text)}</p>`,[{label:tr('取消','Cancel'),value:false},{label:tr('确认','Confirm'),value:true,primary:true}]);
function warnSkippedProjects(projects){if(projects.invalidProjects?.length)toast(tr(`已跳过 ${projects.invalidProjects.length} 个损坏项目，原始数据仍保留。`,`Skipped ${projects.invalidProjects.length} damaged projects. Their original data was retained.`));}
async function projectOperation(operation){
 if(busy||training.running)throw Object.assign(new Error('BUSY'),{code:'BUSY'});busy=true;updateNav();
 try{return await operation();}finally{busy=false;await render();}
}
async function selectProject(next,nextCheckpoint){
 if(nextCheckpoint)validateCheckpoint(nextCheckpoint);
 await api.dispose();project=next;checkpoint=nextCheckpoint;checkpointStorageError=null;
 training={running:false,status:'idle',progress:null,result:null};testImage=testResult=null;imageIndex=activeClass=0;step='images';location.hash='images';saveState='saved';
}
async function exportBackup(){const blob=await packProject(project,getBlob,checkpoint);downloadBlob(blob,`${cleanName(project.title,'project')}.lyproj`);toast(tr('完整项目备份已生成，请妥善保存下载文件。','A full project backup was generated. Keep the downloaded file safe.'));}
async function restoreBackup(files){
 if(!files[0]||busy||training.running)return;busy=true;updateNav();
 try{
  await saves.flush();await checkpointPersist;
  const next=await unpackProject(files[0]),thumbs=new Map();
  for(const image of next.project.images)thumbs.set(image.id,await makeThumbnail(next.images.get(image.id)));
  await storage.importProjectAtomic(next.project,next.images,thumbs,next.checkpoint);await api.dispose();
  project=next.project;checkpoint=next.checkpoint;checkpointStorageError=null;imageIndex=activeClass=0;
  training={running:false,status:'idle',progress:null,result:null};testImage=testResult=null;step='images';location.hash='images';saveState='saved';
  toast(tr('已恢复为一个新项目，原有项目没有被覆盖。','Restored as a new project. Existing projects were not overwritten.'));
 }finally{busy=false;await render();}
}
async function projectMenu(){
 if(busy||training.running){toast(tr('请等待当前操作完成，或先暂停学习，再管理项目。','Wait for the current operation or pause learning before managing projects.'));return;}
 const action=await dialog(tr('项目','Project'),`<p>${esc(project.title)} · ${project.images.length} ${tr('张图片','pictures')}</p><p class="muted">${tr('图片、标注、配置和模型保存在此浏览器。导出备份可跨设备恢复。','Pictures, annotations, settings, and models are stored in this browser. Export a backup to restore on another device.')}</p>`,[{label:tr('新建','New'),value:'new'},{label:tr('打开项目','Open'),value:'open'},{label:tr('导入备份','Import backup'),value:'import'},{label:tr('导出完整备份','Export full backup'),value:'export',primary:true},{label:tr('导出 YOLO 数据集','Export YOLO dataset'),value:'dataset'},{label:tr('删除此项目','Delete this project'),value:'delete'},{label:tr('关闭','Close'),value:null}]);
 if(action==='export')return exportBackup();
 if(action==='dataset'){const snapshot=structuredClone(project);assignSplits(snapshot);const blob=await exportYoloDataset(snapshot,getBlob);downloadBlob(blob,'local-yolo-dataset.zip');return;}
 if(action==='new')return projectOperation(async()=>{await saves.flush();await checkpointPersist;await selectProject(createProject({title:tr('我的项目','My project'),className:tr('我的物体','My object')}),null);stamp();});
 if(action==='open')return projectOperation(async()=>{
  await saves.flush();await checkpointPersist;const projects=await storage.listProjects();warnSkippedProjects(projects);const id=await dialog(tr('打开项目','Open project'),`<div class="stack">${projects.map(item=>`<label class="checkbox-row"><input type="radio" name="open-project" value="${esc(item.id)}" ${item.id===project.id?'checked':''}><span>${esc(item.title)} · ${item.images.length}</span></label>`).join('')}</div>`,[{label:tr('取消','Cancel'),value:null},{label:tr('打开','Open'),value:true,primary:true}],node=>node.querySelector('[name="open-project"]:checked')?.value);
  if(id){const next=validateProject(await storage.getProject(id)),nextCheckpoint=await storage.getCheckpoint(id)||null;await selectProject(next,nextCheckpoint);}
 });
 if(action==='import')return chooseFile({accept:'.lyproj,.zip',onFiles:restoreBackup});
 if(action==='delete'&&await confirmDialog(tr('永久删除此浏览器中的项目？','Permanently delete this browser project?'),tr('将删除此项目的图片、标注和模型。请先导出完整备份。这个操作无法撤销。','This deletes this project’s pictures, annotations, and models. Export a full backup first. This cannot be undone.')))return projectOperation(async()=>{await saves.flush();await checkpointPersist;await storage.deleteProjectConfirmed(project);await selectProject(createProject({title:tr('我的项目','My project'),className:tr('我的物体','My object')}),null);stamp();});
}
async function help(){await dialog(tr('用自己的图片教 AI','Teach AI with your own pictures'),`<div class="stack"><p>${tr('1. 添加不同角度、光照和背景的图片。','1. Add pictures with different angles, lighting, and backgrounds.')}</p><p>${tr('2. 每个希望识别的物体画一个框，让模型知道“看哪里”和“这是什么”。','2. Draw a box around each object so the model knows where to look and what it is.')}</p><p>${tr('3. 开始学习，然后用新图片测试。效果不好时，补充多样图片并继续标注。','3. Start learning, then test a new picture. If results are poor, add varied pictures and mark them.')}</p><p>${tr('4. 保存训练检查点可继续学习；导出 ONNX 可在推理工具中使用。项目备份包含图片和标注。','4. Save a checkpoint to continue learning; export ONNX to use the inference tool. Project backups include pictures and annotations.')}</p><p class="muted">${tr('所有训练都在你自己的设备中运行。计算框架和预训练特征从本站下载，图片不会上传。浏览器本地存储可能被清理，重要项目请导出备份。','Training runs on your device. Frameworks and pretrained features download from this site; pictures are not uploaded. Browser storage can be cleared; export backups for important projects.')}</p></div>`,[{label:tr('开始使用 / 跳过','Start / skip'),value:true,primary:true}]);}

$('#language-toggle').addEventListener('click',()=>setLanguage(language()==='zh'?'en':'zh'));
$('#theme-toggle').addEventListener('click',()=>{cycleTheme();$('#theme-toggle').title=tr(`外观：${theme()}`,`Appearance: ${theme()}`);});
$('#project-menu').addEventListener('click',safe(projectMenu));$('#help').addEventListener('click',safe(help));
nav.addEventListener('click',safe(event=>{const button=event.target.closest('[data-step]');if(button&&!button.disabled)return navigate(button.dataset.step);}));
window.addEventListener('languagechange',()=>render());
window.addEventListener('beforeunload',event=>{if(saves.dirty||training.running){event.preventDefault();event.returnValue='';}});
document.addEventListener('visibilitychange',()=>{if(document.hidden){saves.flush().catch(()=>{});if(training.running)api.call('pause').catch(()=>{});}});
window.addEventListener('hashchange',()=>{const next=location.hash.slice(1);if(['images','annotate','train','test'].includes(next)&&next!==step)navigate(next);});
async function boot(){
 try{const projects=await storage.listProjects();warnSkippedProjects(projects);project=projects[0]?validateProject(projects[0]):null;if(project){checkpoint=await storage.getCheckpoint(project.id)||null;saveState='saved';}}catch(error){storageError=error;}
 if(!project){project=createProject({title:tr('我的项目','My project'),className:tr('我的物体','My object')});stamp();}
 const requested=location.hash.slice(1);if(['images','annotate','train','test'].includes(requested))step=requested;
 await render();
}
boot().catch(error=>{app.textContent=tr('页面无法初始化。请使用支持 Web Workers 和 IndexedDB 的现代浏览器。','The page could not start. Use a modern browser with Web Workers and IndexedDB.');toast(message(error));});
