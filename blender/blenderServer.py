import bpy
import json
import math
import gc
import random
import asyncio
from asyncio import coroutine, sleep, Task, wait_for
import sys
import os
sys.path.append(os.path.dirname(__file__))
import aiohttp
import asyncio_bridge
import feedparser
from asyncio_bridge import BlenderListener
from mathutils import Vector

global IND
IND = 0

def center():
    for i in range(56):
        obj = bpy.data.objects[2+i]
        bpy.ops.object.select_all(action='DESELECT')
        obj.select = True
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY',center='BOUNDS')
        bpy.data.objects[2+i].location.x = 0 
        bpy.data.objects[2+i].location.y = 0
        bpy.data.objects[2+i].location.z = 0

def augment():
    global IND
    IND = IND + 1
    r1 = halton(IND,2)
    r2 = halton(IND,3)
    r3 = halton(IND,5)
    for i in range(56):
        obj = bpy.data.objects[2+i]
        bpy.ops.object.select_all(action='DESELECT')
        obj.select = True
        bpy.ops.transform.rotate(value = (3.14/2)*(2*r1-1),axis = (0.0,0.0,1.0))
        bpy.ops.transform.rotate(value = (3.14/2)*(2*r2-1),axis = (1.0,0.0,0.0))
        bpy.ops.transform.rotate(value = (3.14/2)*(2*r3-1),axis = (0.0,1.0,0.0))
        s1 = 1 + 0.3*(2*random.random()-1)
        s3 = 1 + 0.3*(2*random.random()-1)
        s2 = 1 + 0.3*(2*random.random()-1)
        obj.scale = [s1,s2,s3]

def halton(index,base):
    result = 0
    i = index
    f = 1
    while (i > 0):
        f = f/float(base)
        result = result+f*(i%base)
        i = math.floor(i/float(base))
    return result


def get_view(area,a,b,z,z_look,index,n):
    r = 0.4+0.75*(1+a)
    theta = math.pi*b
    x = r*math.cos(theta)
    y = r*math.sin(theta)
    moveCamera(area,x,y,z,0.5*z_look)
    do_render(area,index,n)

def do_render(area,index,n):
    bpy.context.scene.render.filepath='/ramcache/renders/m'+str(n)+'.png'
    obj = bpy.data.objects['m'+str(index)]
    bpy.ops.object.select_all(action='DESELECT')
    obj.select = True
    bpy.data.objects['Lamp'].select = True
    ctx=bpy.context.copy()
    ctx['area']=area
    bpy.ops.object.hide_render_clear(ctx)       
    bpy.ops.render.render(use_viewport=True,write_still=True)
    bpy.ops.object.hide_render_set(ctx)

def look_at(obj_camera, point):
    loc_camera = obj_camera.matrix_world.to_translation()

    direction = point - loc_camera
    # point the cameras '-Z' and use its 'Y' as up
    rot_quat = direction.to_track_quat('-Z', 'Y')

    # assume we're using euler rotation
    obj_camera.rotation_euler = rot_quat.to_euler()

def moveCamera(area,x,y,z,z_look):
    bpy.ops.object.select_all(action='DESELECT')
    bpy.data.objects['Camera'].select = True
    ctx=bpy.context.copy()
    ctx['area']=area
    bpy.ops.transform.translate(ctx,value=Vector((x,y,z)) - bpy.data.objects['Camera'].location)
    look_at(bpy.data.objects['Camera'],Vector((0.0,0.0,z_look)))

def deleteObjects():
    #bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_by_type(type='MESH')
    bpy.ops.object.delete(use_global=False)

    for item in bpy.data.meshes:
        bpy.data.meshes.remove(item)

    gc.collect()

def loadObjects(batch):
    deleteObjects()

    filepath = "/home/dan/git/shape-classifier/princeton_blend/princeton"
    scn = bpy.context.scene

    with bpy.data.libraries.load(filepath + str(batch) +".blend") as (data_from, data_to):
        data_to.objects = data_from.objects

    for obj in data_to.objects:
        if obj is not None and obj.type != 'CAMERA' and obj.type != 'LAMP':
            scn.objects.link(obj)

    for obj in bpy.data.objects:
        if obj.type == 'CAMERA' or obj.type == 'LAMP':
            if obj.name != 'Camera' and obj.name != 'Lamp':
                bpy.data.objects.remove(obj)

    center()
    augment()

    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            area.spaces[0].region_3d.view_perspective = 'CAMERA'
            override=bpy.context.copy()
            override['area']=area
            bpy.ops.object.select_all(action='DESELECT')
           
            bpy.data.objects['Lamp'].select = True
            bpy.ops.object.hide_render_set(override,unselected=True)
            bpy.data.objects['Lamp'].select = False
    

def callMethod(method,params,area):
    if method == "moveCamera":
        moveCamera(area,params[0],params[1],params[2])
    elif method == "get_view":
        get_view(area,params[0],params[1],params[2],params[3],params[4],params[5])
    elif method == "loadObjects":
        loadObjects(params[0])

@coroutine
def http_server():
    import aiohttp
    from aiohttp import web

    ctx_3d = {}

    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            ctx_3d = area

    bpy.context.scene.render.resolution_x = 128
    bpy.context.scene.render.resolution_y = 128
    bpy.context.scene.render.image_settings.file_format='PNG'
    bpy.context.scene.render.image_settings.color_mode='BW'

    world = bpy.context.scene.world
    wset = world.light_settings
    wset.use_environment_light = True
    wset.environment_energy = 0.2
    wset.gather_method = 'APPROXIMATE'

    @coroutine
    def handle(request):
        data = yield from request.text()
        obj = json.loads(data)
        callMethod(obj['method'],obj['params'],ctx_3d)
        return web.Response(text=json.dumps({'jsonrpc':'2.0','id':obj['id'],'result':'GOOD WORK!'}))


    @coroutine
    def init(loop):
        app = web.Application(loop=loop)
        app.router.add_route('POST', '/', handle)

        srv = yield from loop.create_server(app.make_handler(),
                                            '127.0.0.1', 9090)
        return srv
    yield from init(asyncio.get_event_loop())

if __name__ == "__main__":
    asyncio_bridge.register()
    bpy.ops.bpy.start_asyncio_bridge()
    Task(http_server())
