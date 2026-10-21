/* Template/render smoke test using Node's VM and a minimal document adapter.
   This does not automate a browser or verify layout. Start python -m webapp first. */
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const base = 'http://127.0.0.1:5000';

async function main() {
  const response = await fetch(base+'/api/state');
  assert.equal(response.status,200);
  const snapshot = await response.json();
  const source = fs.readFileSync('webapp/static/app.js','utf8');
  const body = fs.readFileSync('webapp/static/mission.js','utf8')+'\n'+source.slice(0,source.indexOf('refresh().catch(error=>'));
  assert(body.length>10000);
  const results=[];
  for(const route of ['/','/datasets','/models','/training','/tags','/missions','/jobs','/live','/files?path=missions','/settings']) {
    const url = new URL(route,base);
    const nodes = new Map();
    const node = selector => {
      if(!nodes.has(selector)) nodes.set(selector,{innerHTML:'',textContent:'',content:'test-token',className:'',hidden:false,value:'',scrollHeight:0,dataset:{},checked:true,
        classList:{toggle(){}},querySelectorAll:()=>[],addEventListener(){},showModal(){this.open=true;},close(){this.open=false;}});
      return nodes.get(selector);
    };
    const document={hidden:false,querySelector:s=>s==='#mission-form' ? null : node(s),querySelectorAll:()=>[],addEventListener(){},title:''};
    const context=vm.createContext({document,location:{pathname:url.pathname,search:url.search},
      URLSearchParams,encodeURIComponent,FormData,setInterval:()=>0,clearTimeout,setTimeout,
      localStorage:{getItem:()=>null,setItem(){},removeItem(){}},fetch:(path,options)=>fetch(new URL(path,base),options),snapshot,console});
    vm.runInContext(body+'\nstate=snapshot; globalThis.runRender=render; globalThis.testEscape=esc;',context);
    await context.runRender();
    const html=node('#content').innerHTML;
    assert(html.includes('<h1>'),route+' missing heading');
    assert(!html.includes('undefined'),route+' unresolved data');
    assert(!html.includes('Could not open'),route+' failed rendering');
    if(url.pathname==='/missions') {
      assert(html.includes('name="object_provider"'));
      assert(html.includes('value="hsv"'));
      assert(html.includes('name="steering_controller"'));
      assert(html.includes('value="pid"'));
      assert(html.includes('value="kalman"'));
      assert((html.match(/class="field-help"/g)||[]).length>=snapshot.config_schema.length-1);
    }
    assert.equal(context.testEscape('<script>"&'), '&lt;script&gt;&quot;&amp;');
    if(url.pathname==='/live' && snapshot.jobs.some(j=>j.kind==='mission')) {
      const detail=node('#job-detail').innerHTML;
      assert(detail.includes('Live camera'));
      assert(detail.includes('Movement requests'));
      assert(detail.includes('Download full log'));
      assert(detail.includes('id="color-pick-image"'));
      fs.writeFileSync('.control/ui-check/mission-detail.html',detail);
    }
    if(url.pathname==='/jobs') assert(html.indexOf('id="job-detail"')<html.indexOf('id="job-table"'));
    results.push({route,characters:html.length,forms:(html.match(/<form /g)||[]).length});
    fs.mkdirSync('.control/ui-check',{recursive:true});
    fs.writeFileSync('.control/ui-check/'+(url.pathname.slice(1)||'home')+'.html',html);
  }
  fs.writeFileSync('.control/webapp-ui-check.json',JSON.stringify(results,null,2));
  console.log('Rendered all 10 page templates successfully (nonvisual check).');
  console.log(JSON.stringify(results,null,2));
}
main().catch(error=>{console.error(error);process.exitCode=1;});
