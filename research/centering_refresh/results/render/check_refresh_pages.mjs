import fs from 'node:fs/promises';
import http from 'node:http';
import path from 'node:path';
import {spawn} from 'node:child_process';
import {once} from 'node:events';
const root='/home/n/scratch/kb-agent-tmp/BayesianRegressionModels-docs-adaptive-centering';
const output=path.join(root,process.argv[2]||'centering-refresh-browser-v1'); await fs.mkdir(output,{recursive:true});
const server=http.createServer(async(req,res)=>{
 try {let name=decodeURIComponent(new URL(req.url,'http://localhost').pathname);if(!path.extname(name))name+='.html';
 const file=path.join(process.env.PREVIEW_DIST||path.join(root,'centering-refresh-render-v1/build/1'),name);const body=await fs.readFile(file);
 const mime={'.html':'text/html','.js':'text/javascript','.css':'text/css','.png':'image/png','.svg':'image/svg+xml','.woff2':'font/woff2','.json':'application/json'}[path.extname(file)]||'application/octet-stream';
 res.writeHead(200,{'Content-Type':mime});res.end(body);
 }catch{res.writeHead(404);res.end('not found')}
});server.listen(0,'127.0.0.1');await once(server,'listening');const origin=`http://127.0.0.1:${server.address().port}`;
const profile=await fs.mkdtemp('/tmp/kb-cdp-profile-');
const chrome=spawn('/home/n/.local/bin/google-chrome',['--headless=new','--no-sandbox','--disable-gpu','--hide-scrollbars',`--user-data-dir=${profile}`,'--remote-debugging-port=0','about:blank'],{stdio:['ignore','ignore','pipe']});
let log='';chrome.stderr.on('data',d=>{log+=d});
const sleep=ms=>new Promise(r=>setTimeout(r,ms));let browser;
function connection(url){return new Promise((resolve,reject)=>{const ws=new WebSocket(url);let id=0;const pending=new Map();const events=[];
ws.addEventListener('open',()=>resolve({events,send:(method,params={})=>new Promise((resolve,reject)=>{const i=++id;pending.set(i,{resolve,reject});ws.send(JSON.stringify({id:i,method,params}))}),close:()=>ws.close()}));
ws.addEventListener('error',reject);ws.addEventListener('message',({data})=>{const m=JSON.parse(data);if(m.id){const p=pending.get(m.id);pending.delete(m.id);m.error?p.reject(m.error):p.resolve(m.result)}else events.push(m)});
})}

const results=[];
const pages=['adaptive-centering','eight-schools-centering','radon-centering','pupil-centering'];
try{
 let port;for(let i=0;i<100;i++){try{port=(await fs.readFile(path.join(profile,'DevToolsActivePort'),'utf8')).split('\n')[0];break}catch{await sleep(100)}}if(!port)throw Error('Chrome never started');
 const base=`http://127.0.0.1:${port}`;browser=await connection((await(await fetch(base+'/json/version')).json()).webSocketDebuggerUrl);
 const tab=await(await fetch(base+'/json/new?about:blank',{method:'PUT'})).json();const c=await connection(tab.webSocketDebuggerUrl);
 await c.send('Page.enable');await c.send('Runtime.enable');
 const evaluate=async expression=>{const r=await c.send('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value};
 for(const page of pages){
  await c.send('Emulation.setDeviceMetricsOverride',{width:1440,height:1100,deviceScaleFactor:1,mobile:false});
  await c.send('Page.navigate',{url:origin+'/'+page});await sleep(1200);
  await evaluate('Promise.all([...document.images].map(i=>i.decode().catch(()=>null)))');
  const before=await evaluate(`({title:document.title,
   panes:[...document.querySelectorAll('.backend-comparison__tab')].map(x=>x.textContent),
   brokenImages:[...document.images].filter(x=>!x.complete||x.naturalWidth===0).map(x=>x.src),
   mathErrors:[...document.querySelectorAll('[data-mjx-error], [data-mml-node="merror"]')].map(x=>x.textContent),
   displayMath:document.querySelectorAll('mjx-container[display="true"]').length,
   efficiencyTables:[...document.querySelectorAll('.vp-doc table')].filter(t=>t.querySelector('thead')?.textContent.includes('Total gradients')).map(t=>t.querySelectorAll('tbody tr').length),
   sidebar:document.querySelector('.VPSidebar')?.textContent,
   width:innerWidth,scrollWidth:document.documentElement.scrollWidth})`);
  if(before.brokenImages.length||before.mathErrors.length)throw Error(JSON.stringify({page,before}));
  if(JSON.stringify(before.efficiencyTables)!==JSON.stringify([page==='pupil-centering'?15:6]))throw Error(JSON.stringify({page,before}));
  if(page==='pupil-centering'&&before.displayMath!==16)throw Error('Pupil display equations missing');
  if(!before.sidebar.includes('Adaptive centering')||!before.sidebar.includes('Pupil: exact marginalization'))throw Error('Category missing');
  if(page!=='pupil-centering'&&before.panes.length!==4)throw Error('Backend panes missing');
  if(page!=='pupil-centering')await evaluate('document.querySelectorAll(".backend-comparison__tab")[3].click()');
  if(page==='pupil-centering'){await evaluate("document.querySelector('mjx-container[display=\"true\"]').scrollIntoView({block:'center'})");await fs.writeFile(path.join(output,'pupil-equations.png'),Buffer.from((await c.send('Page.captureScreenshot')).data,'base64'));}
  const selected=await evaluate('[...document.querySelectorAll(".backend-comparison__panel:not([hidden])")].map(x=>x.dataset.backendPane)');
  await evaluate(`document.querySelector('img[src*="efficiency"]').scrollIntoView({block:'center'})`);await sleep(100);
  await fs.writeFile(path.join(output,page+'-efficiency.png'),Buffer.from((await c.send('Page.captureScreenshot')).data,'base64'));
  await c.send('Emulation.setDeviceMetricsOverride',{width:390,height:950,deviceScaleFactor:1,mobile:true});await sleep(150);
  await evaluate(`document.querySelector('img[src*="efficiency"]').scrollIntoView({block:'start'})`);
  const mobile=await evaluate('({width:innerWidth,scrollWidth:document.documentElement.scrollWidth})');
  if(mobile.scrollWidth>mobile.width)throw Error(JSON.stringify({page,mobile}));
  await fs.writeFile(path.join(output,page+'-mobile.png'),Buffer.from((await c.send('Page.captureScreenshot')).data,'base64'));
  results.push({page,before,selected,mobile,exceptions:c.events.filter(x=>x.method==='Runtime.exceptionThrown')});c.events.length=0;
 }
 await c.send('Emulation.setDeviceMetricsOverride',{width:1440,height:1100,deviceScaleFactor:1,mobile:false});
 for(const page of pages){
  await evaluate(`document.querySelector('.VPSidebar a[href="/${page}"]').click()`);await sleep(350);
  const state=await evaluate('({url:location.pathname,panes:document.querySelectorAll(".backend-comparison__tab").length})');
  if(state.url!==`/${page}`||state.panes!==(page==='pupil-centering'?0:4))throw Error(JSON.stringify(state));
 }
 const navigationExceptions=c.events.filter(x=>x.method==='Runtime.exceptionThrown');
 if(navigationExceptions.length||results.some(x=>x.exceptions.length))throw Error('Browser exception');
 c.close();await fs.writeFile(path.join(output,'results.json'),JSON.stringify(results,null,2));
 console.log(JSON.stringify(results.map(r=>({page:r.page,efficiencyRows:r.before.efficiencyTables,displayMath:r.before.displayMath,brokenImages:r.before.brokenImages,selected:r.selected,mobile:r.mobile}))));
 console.log('FOUR_PAGE_BROWSER_COMPLETE');
}finally{
 if(browser){await browser.send('Browser.close').catch(()=>{});browser.close()}
 if(chrome.exitCode===null){await Promise.race([once(chrome,'exit'),sleep(1500)]);if(chrome.exitCode===null){chrome.kill();await once(chrome,'exit').catch(()=>{})}}
 server.close();await fs.rm(profile,{recursive:true,force:true});await fs.writeFile(path.join(output,'chrome.log'),log);
}
