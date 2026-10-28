const vm=require('node:vm'),fs=require('node:fs'),assert=require('node:assert/strict');
const source=fs.readFileSync('k1live/web/display.js','utf8');
const code=source.slice(source.indexOf('// ---------------------------------------------------------------- live images'),source.indexOf('// ---------------------------------------------------------------- fallback video'));
let fetches=[],decodes=[],rafs=[],timers=[],revoked=[];
const cam={id:'cam',src:'old-cam',removeAttribute(){this.src='';}},robot={id:'robot',src:'old-robot',removeAttribute(){this.src='';}};
const c={streamsOn:false,preview:false,cam,robot,performance:{now:()=>0},Date,AbortController,
 setTimeout:(fn,delay)=>{timers.push({fn,delay});return timers.length;},clearTimeout:()=>{},
 requestAnimationFrame:fn=>rafs.push(fn),URL:{createObjectURL:b=>'blob:'+b,revokeObjectURL:u=>revoked.push(u)},
 fetch:(url,opts)=>new Promise(resolve=>fetches.push({url,opts,resolve})),
 Image:class {decode(){return new Promise(resolve=>decodes.push(resolve));}}};
vm.createContext(c);vm.runInContext(code,c);
async function flush(){for(let i=0;i<12;i++)await Promise.resolve();}
(async()=>{
 c.startStreams();c.startStreams();assert.equal(fetches.length,2,'only one in-flight pair');
 fetches[0].resolve({ok:true,blob:async()=> 'cam'});await flush();decodes[0]();await flush();
 assert.equal(cam.src,'old-cam');assert.equal(robot.src,'old-robot');
 fetches[1].resolve({ok:true,blob:async()=> 'robot'});await flush();decodes[1]();await flush();
 assert.equal(cam.src,'old-cam');assert.equal(rafs.length,1);
 rafs[0]();await flush();assert.equal(cam.src,'blob:cam');assert.equal(robot.src,'blob:robot');
 c.stopStreams();assert.equal(cam.src,'');assert.equal(robot.src,'');assert.equal(revoked.length,2);
 c.startStreams();timers.at(-1).fn();assert.equal(fetches.length,4);
 fetches[2].resolve({ok:false});await flush();
 fetches[3].resolve({ok:true,blob:async()=> 'late-robot'});await flush();decodes[2]();await flush();
 assert.equal(cam.src,'');assert.equal(robot.src,'');
 assert.ok(revoked.includes('blob:late-robot'),'failed pairs clean up even a late successful image');
 c.stopStreams();
 console.log('PASS: one in-flight pair, atomic paint, stop and failed-pair URL cleanup');
})().catch(e=>{console.error(e);process.exitCode=1});
