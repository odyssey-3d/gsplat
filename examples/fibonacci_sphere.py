import torch
from utils import knn

class FibonacciSphere(torch.nn.Module):
    def __init__(self, radius: torch.Tensor, samples, randomize=True):
        super().__init__()
        self.radius = radius
        self.samples = samples
        self.randomize = randomize

        # Bounding skybox sphere
        self.device = "cuda"
        self._sb_xyz = torch.empty(0, device=self.device)
        self._sb_features_dc = torch.empty(0, device=self.device)
        self._sb_features_rest = torch.empty(0, device=self.device)
        self._sb_scaling = torch.empty(0, device=self.device)
        self._sb_rotation = torch.empty(0, device=self.device)
        self._sb_opacity = torch.empty(0, device=self.device)

    def create_fibonacci_sphere(self, randomize=True):
        """
        Generates `samples` number of points distributed approximately equally on a sphere.

        :param randomize: Randomizes point distribution using a random factor.
        :return: Tensor of shape (samples, 3) representing 3D points.
        """

        offset = 2.0 / self.samples
        increment = torch.pi * (3.0 - torch.sqrt(torch.tensor(5.0)))  # Golden angle in radians

        phi = ((torch.arange(self.samples)) % self.samples) * increment

        y = (((torch.arange(self.samples)) * offset) - 1) + (offset / 2)
        r = torch.sqrt(1 - torch.pow(y, torch.tensor(2.0)))
        x = torch.cos(phi) * r
        z = torch.sin(phi) * r

        self._sb_xyz = torch.stack([x, y, z]).T * self.radius

        if randomize:
            random_xyz_jitter = (torch.rand(self._sb_xyz.shape) - 0.5) * (0.01 * self.radius)

            self._sb_xyz = self._sb_xyz + torch.rand(self._sb_xyz.shape) + random_xyz_jitter

        # filter so scene sphere is hemisphere
        self._sb_xyz = self._sb_xyz[z > -0.5]
        self._sb_xyz = self._sb_xyz.cuda()

    def create(self, xyz, shs_dim):
        with torch.no_grad():
            self.create_fibonacci_sphere(randomize=True)

            dist2_avg = (knn(self._sb_xyz, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
            dist_avg = torch.sqrt(dist2_avg)
            self._sb_scaling = torch.log(dist_avg).unsqueeze(-1).repeat(1, 3)  # [N, 3]

            N = self._sb_xyz.shape[0]
            self._sb_rotation = torch.zeros((N, 4), device=self.device)
            self._sb_rotation[:, 0] = 1
            sb_features_dc = torch.zeros((N, 1, 3), dtype=torch.float, device=self.device)
            sb_features_rest = torch.zeros((N, shs_dim - 1, 3), dtype=torch.float, device=self.device)
            self._sb_opacity = torch.ones((N, 1), dtype=torch.float, device=self.device)

            self._sb_features_dc = torch.nn.Parameter(sb_features_dc.requires_grad_(True))
            self._sb_features_rest = torch.nn.Parameter(sb_features_rest.requires_grad_(True))

    @property
    def get_sb_xyz(self):
        return self._sb_xyz

    @property
    def get_sb_scaling_activation(self):
        return torch.exp(self._sb_scaling)

    @property
    def get_sb_rotation_activation(self):
        return torch.nn.functional.normalize(self._sb_rotation)

    @property
    def get_sb_features(self):
        sb_features_dc = self._sb_features_dc
        sb_features_rest = self._sb_features_rest
        return torch.cat((sb_features_dc, sb_features_rest), dim=1)

    @property
    def get_sb_opacity(self):
        return torch.sigmoid(self._sb_opacity)


def fib_sphere_to_paramdict(fib_sphere: FibonacciSphere):
    xyz = fib_sphere.get_sb_xyz
    N = xyz.shape[0]
    device = xyz.device

    w = torch.zeros(N, device=device)
    means = xyz.clone()
    scales_log = fib_sphere._sb_scaling  # or torch.log(...) if stored in linear
    quats = torch.nn.functional.normalize(fib_sphere._sb_rotation, dim=-1)
    raw_opa = fib_sphere.get_sb_opacity.clamp(1e-6, 1-1e-6)
    opacities = torch.logit(raw_opa.squeeze(-1))

    # Spherical harmonics
    sh0 = fib_sphere._sb_features_dc  # [N, 1, 3]
    shN = fib_sphere._sb_features_rest # [N, (sh_degree-1), 3]

    return {
        "means": means,
        "w": w,
        "scales": scales_log,
        "quats": quats,
        "opacities": opacities,
        "sh0": sh0,
        "shN": shN,
    }
