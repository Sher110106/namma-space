#!/usr/bin/env python3
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parents[1]/'docs/tools'))
from chrome import screenshot
src=Path(__file__).parent/'cards/cards.html'; out=Path(__file__).parent/'frames'; out.mkdir(parents=True,exist_ok=True)
for name,title,sub,note in [('title','Namma Space','One phone video. One command. A room you can walk through.','Round 1 technical abstract and browser proof of concept'),('limits','Scope','One indoor room in Round 1.','Search, routes, multiple floors and WebXR AR are not built yet.')]:
    extra=f'<script>document.getElementById("title").textContent={title!r};document.getElementById("sub").textContent={sub!r};document.getElementById("note").textContent={note!r}</script>'
    screenshot(src,out/f'{name}.png',1920,1080,extra_head=extra)
    print(out/f'{name}.png')
