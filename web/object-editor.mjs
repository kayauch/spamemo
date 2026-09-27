let instanceCount=0;

/** A small editor; the caller owns persistence, model focus, and reloading. */
export function installObjectEditor({objects,focus=()=>{},clearFocus=()=>{},save}){
  const stylesheet='/static/object-editor.css?v=20260927-objects1';
  if(!document.querySelector('link[data-object-editor-style]')){
    const link=document.createElement('link');link.rel='stylesheet';link.href=stylesheet;
    link.dataset.objectEditorStyle='true';document.head.append(link);
  }
  const dialog=document.createElement('dialog');dialog.className='object-editor';
  const titleId='object-editor-title-'+(++instanceCount);
  dialog.setAttribute('aria-labelledby',titleId);document.body.append(dialog);
  let selectedId=null,draftName='',confirming=false,busy=false,error='',notice='';
  const key=id=>String(id);
  const rows=()=>{const value=objects();return Array.isArray(value)?value:[];};
  const element=(tag,text,className)=>{
    const node=document.createElement(tag);if(text!=null)node.textContent=text;
    if(className)node.className=className;return node;
  };
  const button=(text,action,className='')=>{
    const node=element('button',text,className);node.type='button';node.addEventListener('click',action);return node;
  };
  function status(){
    const alert=dialog.querySelector('.object-editor-error');
    const progress=dialog.querySelector('.object-editor-status');
    if(alert){alert.textContent=error;alert.hidden=!error;}
    if(progress){progress.textContent=notice;progress.hidden=!notice;}
    for(const control of dialog.querySelectorAll('button,input'))control.disabled=busy;
    dialog.setAttribute('aria-busy',String(busy));
  }
  function close(){if(busy)return;dialog.close();}
  function choose(row){
    selectedId=key(row.id);draftName=row.name||'';confirming=false;error='';notice='';
    render();if(!row.hidden)focus(row.id);
    dialog.querySelector('.object-editor-name')?.focus();
  }
  function showList(){selectedId=null;draftName='';confirming=false;error='';notice='';clearFocus();render();}
  async function persist(row,patch){
    if(busy)return;
    busy=true;error='';notice='保存しています…';status();
    try{
      await save(row.id,patch);
      notice='保存しました。部屋に反映しています…';status();
      // Stay busy until the caller reloads the room; no duplicate writes.
    }catch(reason){
      busy=false;notice='';error=reason?.message||'保存できませんでした。もう一度お試しください。';status();
    }
  }
  function footer(){
    const alert=element('p','','object-editor-error');alert.setAttribute('role','alert');alert.hidden=true;
    const progress=element('p','','object-editor-status');progress.setAttribute('role','status');progress.setAttribute('aria-live','polite');progress.hidden=true;
    dialog.append(alert,progress);status();
  }
  function render(){
    const all=rows(),selected=all.find(row=>key(row.id)===selectedId);
    if(selectedId!=null&&!selected){selectedId=null;confirming=false;clearFocus();}
    dialog.replaceChildren();
    const header=element('header',null,'object-editor-header');
    const heading=element('h2',confirming?'この物を削除しますか？':selected?'物の名前を変更':'認識した物');heading.id=titleId;
    header.append(heading,button('閉じる',close,'object-editor-close'));dialog.append(header);
    if(selected&&confirming){
      dialog.append(element('p',selected.name||'名前のない物','object-editor-selected-name'));
      dialog.append(element('p','部屋の表示と検索から外します。中の物の記録も非表示になります。削除済みの一覧から戻せます。','object-editor-description'));
      const actions=element('div',null,'object-editor-actions');
      const cancel=button('キャンセル',()=>{confirming=false;error='';render();dialog.querySelector('.object-editor-delete')?.focus();},'object-editor-cancel');
      actions.append(cancel,button('削除する',()=>persist(selected,{hidden:true}),'object-editor-danger'));
      dialog.append(actions);footer();queueMicrotask(()=>cancel.focus());return;
    }
    if(selected){
      dialog.append(button('‹ 一覧に戻る',showList,'object-editor-back'));
      dialog.append(element('p',(selected.name||'名前のない物')+' を青く表示しています。','object-editor-description'));
      const form=element('form',null,'object-editor-form');
      const label=element('label','名前');
      const input=element('input',null,'object-editor-name');input.type='text';input.value=draftName;
      input.maxLength=120;input.required=true;input.autocomplete='off';
      input.addEventListener('input',()=>{draftName=input.value;input.setCustomValidity('');});label.append(input);form.append(label);
      const actions=element('div',null,'object-editor-actions');
      const rename=element('button','名前を保存','object-editor-primary');rename.type='submit';
      actions.append(button('この物を削除',()=>{draftName=input.value;confirming=true;error='';render();},'object-editor-delete'),rename);form.append(actions);
      form.addEventListener('submit',event=>{
        event.preventDefault();if(busy)return;
        const name=input.value.trim();input.setCustomValidity(name?'':'名前を入力してください。');
        if(!form.reportValidity())return;draftName=name;input.value=name;persist(selected,{name});
      });dialog.append(form);footer();return;
    }
    dialog.append(element('p','直したい物を選んでください。','object-editor-description'));
    const visible=all.filter(row=>!row.hidden),list=element('div',null,'object-editor-list');
    for(const row of visible){
      const item=button('',()=>choose(row),'object-editor-row');
      item.append(element('span',row.name||'名前のない物'),element('span','›','object-editor-chevron'));list.append(item);
    }
    if(!visible.length)list.append(element('p','表示中の物はありません。','object-editor-empty'));
    dialog.append(list);
    const hidden=all.filter(row=>row.hidden),details=element('details',null,'object-editor-deleted');
    const summary=element('summary','削除済み'+(hidden.length?'（'+hidden.length+'）':''));details.append(summary);
    if(!hidden.length)details.append(element('p','削除した物はありません。','object-editor-empty'));
    for(const row of hidden){
      const item=element('div',null,'object-editor-deleted-row');
      item.append(element('span',row.name||'名前のない物'),button('戻す',()=>persist(row,{hidden:false}),'object-editor-restore'));
      details.append(item);
    }
    dialog.append(details);footer();
  }
  dialog.addEventListener('cancel',event=>{event.preventDefault();if(!busy)close();});
  dialog.addEventListener('close',()=>{clearFocus();selectedId=null;confirming=false;});
  // Disclosure controls are not form controls, so also stop them while saving.
  dialog.addEventListener('click',event=>{if(busy){event.preventDefault();event.stopImmediatePropagation();}},true);
  dialog.addEventListener('keydown',event=>{if(busy&&['Escape','Enter',' '].includes(event.key)){event.preventDefault();event.stopImmediatePropagation();}},true);
  return {
    open(id){
      if(busy)return;
      selectedId=null;confirming=false;error='';notice='';draftName='';
      const selected=id==null?null:rows().find(row=>key(row.id)===key(id)&&!row.hidden);
      if(selected){selectedId=key(selected.id);draftName=selected.name||'';}
      render();if(!dialog.open)dialog.showModal();
      if(selected){focus(selected.id);dialog.querySelector('.object-editor-name')?.focus();}
      else{clearFocus();dialog.querySelector('.object-editor-row, .object-editor-close')?.focus();}
    },close
  };
}
