"""Presentation scenery. Used by the render model only; adds no moving body or robot joint."""
from pathlib import Path
import xml.etree.ElementTree as ET


def snu_scene_xml(xml: str) -> str:
    root = ET.fromstring(xml)
    assets = root.findall('asset')
    asset = assets[-1]
    for texture in root.findall('.//texture'):
        if texture.get('type') == 'skybox':
            # Warm horizon / cool upper sky, matching the supplied FINALFINAL film.
            texture.set('rgb1', '0.66 0.76 0.81')
            texture.set('rgb2', '0.96 0.84 0.70')
        elif texture.get('name') == 'evfloor':
            texture.set('rgb1', '0.72 0.70 0.66')
            texture.set('rgb2', '0.82 0.80 0.75')
            texture.set('markrgb', '0.76 0.74 0.69')
    for material in root.findall('.//material'):
        if material.get('name') == 'evfloor':
            material.set('texrepeat', '1 1')
            material.set('reflectance', '0')
    visual = root.find('visual')
    if visual is None:
        visual = ET.SubElement(root, 'visual')
        ET.SubElement(visual, 'headlight', diffuse='0.25 0.25 0.25', ambient='0.34 0.34 0.34', specular='0 0 0')
        ET.SubElement(visual, 'rgba', haze='0.91 0.84 0.74 1')
    global_cfg = visual.find('global')
    if global_cfg is None:
        global_cfg = ET.SubElement(visual, 'global')
    global_cfg.set('offwidth', '1920')
    global_cfg.set('offheight', '1080')
    haze = root.find('.//visual/rgba')
    if haze is not None:
        haze.set('haze', '0.91 0.84 0.74 1')
    mapcfg = root.find('.//visual/map')
    if mapcfg is None:
        mapcfg = ET.SubElement(visual, 'map')
    mapcfg.set('zfar', '150')
    mesh = Path(__file__).resolve().parent / 'assets/snu_gate.obj'
    if not mesh.is_file():
        raise FileNotFoundError(f'SNU render scenery requires {mesh}')
    ET.SubElement(asset, 'mesh', name='snu_gate_mesh', file=str(mesh), scale='7 7 7')
    ET.SubElement(asset, 'material', name='snu_gate_mat', rgba='0.498 0.533 0.576 1',
                  specular='0.08', shininess='0.05')
    world = root.findall('worldbody')[-1]
    # The supplied asset is Z-up and faces +X. Behind a robot facing +X, toward the front camera.
    ET.SubElement(world, 'geom', name='snu_gate_visual', type='mesh', mesh='snu_gate_mesh',
                  material='snu_gate_mat', pos='-13 0 0', contype='0', conaffinity='0', group='2',
                  density='0')
    for geom in root.findall('.//geom'):
        if geom.get('name') == 'floor':
            geom.set('size', '100 100 0.1')
    return ET.tostring(root, encoding='unicode')


def color_snu_sky(model):
    """Procedural cubemap: beige at the horizon, pale blue above, as in the film."""
    import numpy as np
    import mujoco

    for i in range(model.ntex):
        if model.tex_type[i] != mujoco.mjtTexture.mjTEXTURE_SKYBOX:
            continue
        w, h = int(model.tex_width[i]), int(model.tex_height[i]) // 6
        if w != h or int(model.tex_nchannel[i]) != 3:
            raise ValueError('SNU scenery expects a square RGB sky cubemap')
        u, v = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
        norm = np.sqrt(1 + u*u + v*v)
        faces = []
        for face in range(6):
            z = (np.ones_like(v) if face == 2 else -np.ones_like(v) if face == 3 else -v) / norm
            blend = np.clip(z / 0.45, 0, 1)
            blend = blend*blend*(3 - 2*blend)
            horizon = np.array([0.96, 0.85, 0.72])
            blue = np.array([0.70, 0.80, 0.85])
            faces.append(np.uint8(np.clip((horizon + blend[..., None]*(blue-horizon))*255, 0, 255)))
        pixels = np.stack(faces).ravel()
        adr = int(model.tex_adr[i])
        model.tex_data[adr:adr+len(pixels)] = pixels
