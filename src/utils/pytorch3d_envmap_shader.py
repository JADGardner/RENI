import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch3d.structures import Meshes
from pytorch3d.renderer import Materials, TensorProperties
from pytorch3d.common import Device
from pytorch3d.renderer.utils import TensorProperties

from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.renderer.mesh.rasterizer import Fragments
from pytorch3d.io import load_obj
from pytorch3d.transforms import RotateAxisAngle
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    TexturesVertex,
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    TexturesVertex,
    Materials,
)

import numpy as np

from typing import Optional


class EnvironmentMap:
    def __init__(
        self,
        environment_map: torch.Tensor = None,
        directions: torch.Tensor = None,
        sineweight: torch.Tensor = None,
    ) -> None:
        self.directions = directions
        self.environment_map = environment_map * sineweight
        self.environment_map = self.environment_map.squeeze(0)


def blinn_phong_shading_env_map(
    device, meshes, fragments, envmap, cameras, materials, kd, ks
) -> torch.Tensor:
    """
    Apply per pixel shading. First interpolate the vertex normals and
    vertex coordinates using the barycentric coordinates to get the position
    and normal at each pixel. Then compute the illumination for each pixel.
    Args:
        meshes: Batch of meshes
        fragments: Fragments named tuple with the outputs of rasterization
        envmap: lighting colours and lighting directions
        cameras: Cameras class containing a batch of cameras
        materials: Materials class containing a batch of material properties
    Returns:
        colors: (H, W, 3)
    """
    verts = meshes.verts_packed()  # (V, 3)
    faces = meshes.faces_packed()  # (F, 3)
    vertex_normals = meshes.verts_normals_packed()  # (V, 3)
    faces_verts = verts[faces]
    faces_normals = vertex_normals[faces]
    pixel_directions = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, faces_verts
    )  # (N, ..., 3) xyz coordinates of the points.
    pixel_normals = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, faces_normals
    )  # (N, ..., 3) xyz normal vectors for each point.

    light_directions = envmap.directions.to(
        device=device
    )  # (J, 3) unit vector associated with the direction of each pixel in a panoramic image
    light_colors = envmap.environment_map.to(
        device=device
    )  # (J, 3) RGB color of the environment map.

    camera_position = cameras.get_camera_center().squeeze()
    shininess = materials.shininess.to(device=device)
    pixel_normals = (
        pixel_normals.squeeze()
    )  # from (B, H, W, K, 3) -> (H, W, 3) assume B (batch) and K = 1 for now
    pixel_normals = F.normalize(pixel_normals, p=2, dim=-1, eps=1e-6)
    # create copies of light directions for batch matrix multiplication
    L_batch = light_directions.repeat(pixel_normals.shape[0], 1, 1)  # (H, J, 3)
    L_batch = torch.permute(L_batch, (0, 2, 1))  # (H, 3, J)
    # dot product between every image pixel and every light direction
    diffuse = torch.einsum("bij,bjk->bik", pixel_normals, L_batch)  # (H, W, J)
    diffuse = torch.clamp(diffuse, min=0.0, max=1.0)
    # scale every dot product by colour of light source, prescaled by sineweight
    # model_output should be of shape (H*W, 3)
    diffuse = torch.einsum("ij,kli->klj", light_colors, diffuse)  # (H, W, 3)

    # create half-way vectors
    view_direction = (camera_position - pixel_directions).squeeze()
    view_direction = F.normalize(view_direction, p=2, dim=-1, eps=1e-6)
    view_direction_batch = view_direction.repeat(
        light_directions.shape[0], 1, 1, 1
    )  # (J, H, W, 3)
    view_direction_batch = torch.permute(
        view_direction_batch, (1, 2, 0, 3)
    )  # (H, W, J, 3)
    # Half-way vectors between every pixels view-direction and all 'J' light directions
    H = view_direction_batch + light_directions  # (H, W, J, 3)
    H = F.normalize(H, p=2, dim=-1, eps=1e-6)
    # dot product between every image pixel normal and every half-way vector
    specular = torch.einsum("mnj,mnkj->mnk", pixel_normals, H)  # (H, W, J)
    specular = torch.clamp(specular, min=0.0, max=1.0)
    specular = torch.pow(specular, shininess)
    # scale every dot product by colour of light source, prescaled by sineweight
    # model_output should be of shape (H*W, 3)
    specular = torch.einsum("ij,kli->klj", light_colors, specular)  # (H, W, 3)
    bp_specular_normalisation_factor = (shininess + 2) / (
        4 * (2 - torch.exp(-shininess / 2))
    )
    colors = kd * diffuse + bp_specular_normalisation_factor * ks * specular

    return colors, pixel_normals


class BlinnPhongShaderEnvMap(nn.Module):
    """
    Per pixel lighting - the lighting model is applied using the interpolated
    coordinates and normals for each pixel. The blending function returns the
    soft aggregated color using all the faces per pixel.
    """

    def __init__(
        self,
        device: Device = "cpu",
        cameras: Optional[TensorProperties] = None,
        envmap: EnvironmentMap = None,
        materials: Optional[Materials] = None,
        kd=None,
        ks=None,
    ) -> None:
        super().__init__()
        self.envmap = envmap
        self.materials = (
            materials if materials is not None else Materials(device=device)
        )
        self.cameras = cameras
        self.device = device
        self.kd = kd
        self.ks = ks

    def to(self, device: Device):
        # Manually move to device modules which are not subclasses of nn.Module
        cameras = self.cameras
        if cameras is not None:
            self.cameras = cameras.to(device)
        self.materials = self.materials.to(device)
        self.envmap = self.envmap.to(device)
        return self

    def forward(
        self, fragments: Fragments, meshes: Meshes, envmap: EnvironmentMap, **kwargs
    ) -> torch.Tensor:
        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                or in the forward pass of BlinnPhongShader"
            raise ValueError(msg)

        # texels = meshes.sample_textures(fragments)
        envmap = envmap
        materials = kwargs.get("materials", self.materials)
        colors, pixel_normals = blinn_phong_shading_env_map(
            device=self.device,
            meshes=meshes,
            fragments=fragments,
            envmap=envmap,
            cameras=cameras,
            materials=materials,
            kd=self.kd,
            ks=self.ks,
        )
        return colors, pixel_normals


def build_renderer(obj_path, obj_rotation, img_size, kd, device):
    # Load obj file
    verts, faces_idx, _ = load_obj(obj_path, load_textures=False, device=device)
    faces = faces_idx.verts_idx
    rot_y_90 = RotateAxisAngle(obj_rotation, "Y", device=device)
    verts = rot_y_90.transform_points(verts)
    verts_rgb = torch.tensor([1.0, 1.0, 1.0])
    verts_rgb = verts_rgb.repeat(verts.shape[0], 1).unsqueeze(0)  # (1, V, 3)
    textures = TexturesVertex(verts_features=verts_rgb.to(device))

    # Create a Meshes object for the bunny. Here we have only one mesh in the batch.
    mesh = Meshes(verts=[verts.to(device)], faces=[faces.to(device)], textures=textures)

    materials = Materials(shininess=500)

    # Initialize a perspective camera.
    cameras = FoVPerspectiveCameras(device=device)

    raster_settings = RasterizationSettings(
        image_size=img_size,
        blur_radius=0.0,
        faces_per_pixel=1,
        perspective_correct=False,
    )

    ks = 1.0 - kd

    blinn_phong_envmap_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=BlinnPhongShaderEnvMap(
            device=device,
            cameras=cameras,
            envmap=None,
            materials=materials,
            kd=kd,
            ks=ks,
        ),
    )
    R, T = look_at_view_transform(2.0, 0.0, 0.0, degrees=True, device=device)
    return blinn_phong_envmap_renderer, R, T, mesh


class NormalMapShader(nn.Module):
    def __init__(
        self, device: Device = "cpu", cameras: Optional[TensorProperties] = None
    ) -> None:
        super().__init__()
        # self.lights = lights if lights is not None else PointLights(device=device)
        self.cameras = cameras
        self.device = device

    # pyre-fixme[14]: `to` overrides method defined in `Module` inconsistently.

    def to(self, device: Device):
        # Manually move to device modules which are not subclasses of nn.Module
        cameras = self.cameras
        if cameras is not None:
            self.cameras = cameras.to(device)
        self.materials = self.materials.to(device)
        return self

    def forward(self, fragments: Fragments, meshes: Meshes, **kwargs) -> torch.Tensor:
        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                or in the forward pass of NormalMapShader"
            raise ValueError(msg)

        faces = meshes.faces_packed()  # (F, 3)
        vertex_normals = meshes.verts_normals_packed()  # (V, 3)
        faces_normals = vertex_normals[faces]
        pixel_normals = interpolate_face_attributes(
            fragments.pix_to_face, fragments.bary_coords, faces_normals
        )  # (N, ..., 3) xyz normal vectors for each point.
        pixel_normals = (
            pixel_normals.squeeze()
        )  # from (B, H, W, K, 3) -> (H, W, 3) assume B and K = 1
        return pixel_normals


def get_normal_map(obj_path, obj_rotation, device):
    # Load obj file
    verts, faces_idx, _ = load_obj(obj_path, load_textures=False, device=device)
    rot_y_90 = RotateAxisAngle(obj_rotation, "Y", device=device)
    verts = rot_y_90.transform_points(verts)
    faces = faces_idx.verts_idx
    # verts_rgb = torch.ones_like(verts)[None] # (1, V, 3)
    verts_rgb = torch.tensor([1.0, 1.0, 1.0])
    verts_rgb = verts_rgb.repeat(verts.shape[0], 1).unsqueeze(0)  # (1, V, 3)
    textures = TexturesVertex(verts_features=verts_rgb.to(device))

    # Create a Meshes object for the bunny. Here we have only one mesh in the batch.
    mesh = Meshes(verts=[verts.to(device)], faces=[faces.to(device)], textures=textures)

    # Initialize a perspective camera.
    cameras = FoVPerspectiveCameras(device=device)

    raster_settings = RasterizationSettings(
        image_size=128, blur_radius=0.0, faces_per_pixel=1, perspective_correct=False
    )

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=NormalMapShader(device=device, cameras=cameras),
    )

    # Get the position of the camera based on the spherical angles
    R, T = look_at_view_transform(2.0, 0.0, 0.0, degrees=True, device=device)
    normal_map = renderer(meshes_world=mesh, R=R, T=T)
    normal_map = normal_map.cpu().detach().numpy()
    normal_map[normal_map == 0.0] = np.nan
    return normal_map
