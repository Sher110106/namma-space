#!/usr/bin/env node
// Short recorder for the viewer fixture. Uses the cached Playwright Chromium shell via the repo's CDP helper.
import fs from 'node:fs'; import path from 'node:path'; import { fileURLToPath } from 'node:url';
import { launchChrome, sleep } from '../viewer/test/lib/chrome.mjs';
import { startServer } from '../viewer/test/lib/server.mjs';
const here=path.dirname(fileURLToPath(import.meta.url)); const args=new Map(); for(let i=2;i<process.argv.length;i++){let a=process.argv[i];if(a.startsWith('--'))args.set(a.slice(2),process.argv[i+1]&&!process.argv[i+1].startsWith('--')?process.argv[++i]:'1')}
const web=path.resolve(args.get('web')||'../viewer/dist'), out=path.resolve(args.get('out')||path.join(here,'frames')), seconds=Number(args.get('seconds')||8); fs.mkdirSync(out,{recursive:true});
const srv=await startServer([{prefix:'/',dir:web}],{}); const browser=await launchChrome({width:1280,height:720}); const page=await browser.newPage();
try { await page.goto(`${srv.origin}/?scene=honka_demo&pose=0,1.42,0,0,0&bench=1&keys=1`); await sleep(1500); try{await page.click(640,650)}catch{}; await page.waitFor('window.__ns && window.__ns.firstFrameMs',15000).catch(()=>{});
  const start=Date.now(); let i=0; while(Date.now()-start<seconds*1000){await page.key('KeyW',true); await sleep(220); await page.key('KeyW',false); await page.key('KeyD',true); await sleep(120); await page.key('KeyD',false); await page.screenshot(path.join(out,`tour-${String(i++).padStart(4,'0')}.png`)); await sleep(260)}
  console.log(`saved ${i} frames to ${out}`);
} finally {await page.close(); await browser.close(); await srv.close();}
