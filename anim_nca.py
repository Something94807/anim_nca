"""Animated NCA: grow a sprite from one cell, then loop its animation forever.

Keys: SPACE watch, R restart, T train, Up/Down speed, ESC quit.
Left click erases, right click plants a new seed, mouse wheel sets the eraser size.
"""
import os

import numpy as np
import pygame
import torch
import torch.nn.functional as F
from PIL import Image, ImageSequence

SPRITE = "sonic.gif"
GROW = 48                  # steps to grow into frame 0
FRAME_STEPS = 30           # steps per animation frame
CHANNELS = 24
HIDDEN = 256
FIRE_RATE = 0.5
BATCH = 8
POOL_SIZE = 1024
ROLLOUT = (60, 90)
TRAIN_STEPS = 6000
LR = 2e-3
SCALE = 6
TILE_SCALE = 3
REPORT_EVERY = 250
WATCH_SIZE = 132           # canvas size when watching; training uses the sprite's size
WATCH_SCALE = 4
SPEEDS = [0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64]
WRAP_WHEN_WATCHING = True
SAVE_FILE = "anim_nca_free_c24_256.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_frames():
    frames = []
    for fr in ImageSequence.Iterator(Image.open(SPRITE)):
        img = np.array(fr.convert("RGBA"), dtype=np.float32) / 255
        img[..., :3] *= img[..., 3:]                          # premultiplied alpha
        frames.append(torch.tensor(img).permute(2, 0, 1))
    return torch.stack(frames).to(DEVICE)


def frame_at(age, n_frames):
    return ((age - GROW).clamp(min=0) // FRAME_STEPS) % n_frames


def is_check(age):
    # last step of each frame's slot
    return (age >= GROW) & ((age - GROW) % FRAME_STEPS == FRAME_STEPS - 1)


class NCA(torch.nn.Module):
    def __init__(self):
        super().__init__()
        ident = torch.tensor([[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]])
        sobel = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 8
        kernels = torch.stack([ident, sobel, sobel.T])
        self.register_buffer("kernels", kernels.repeat(CHANNELS, 1, 1)[:, None])
        self.B = torch.nn.Conv2d(3 * CHANNELS, HIDDEN, 1)
        self.C = torch.nn.Conv2d(HIDDEN, CHANNELS, 1)
        torch.nn.init.xavier_uniform_(self.B.weight)
        torch.nn.init.zeros_(self.B.bias)
        torch.nn.init.zeros_(self.C.weight)
        torch.nn.init.zeros_(self.C.bias)
        self.wrap = False

    @staticmethod
    def seed(n, size):
        x = torch.zeros(n, CHANNELS, size, size, device=DEVICE)
        x[:, 3:, size // 2, size // 2] = 1.0
        return x

    def pad(self, x):
        return F.pad(x, (1, 1, 1, 1), mode="circular" if self.wrap else "constant")

    def alive(self, x):
        alpha = x[:, 3:4]
        if self.wrap:
            return F.max_pool2d(F.pad(alpha, (1, 1, 1, 1), mode="circular"), 3, stride=1) > 0.1
        return F.max_pool2d(alpha, 3, stride=1, padding=1) > 0.1

    def step(self, x):
        pre = self.alive(x)
        u = F.conv2d(self.pad(x), self.kernels, groups=CHANNELS)
        fire = (torch.rand_like(x[:, :1]) < FIRE_RATE).float()
        x = x + self.C(F.relu(self.B(u))) * fire
        return x * (pre & self.alive(x)).float()


def to_surface(x, scale):
    x = torch.nan_to_num(x)
    rgb = (1 - x[3:4].clamp(0, 1) + x[:3]).clamp(0, 1)
    img = (rgb.permute(2, 1, 0).detach().cpu().numpy() * 255).astype(np.uint8)
    surf = pygame.surfarray.make_surface(img)
    return pygame.transform.scale(surf, (img.shape[0] * scale, img.shape[1] * scale))


def main():
    frames = load_frames()
    n_frames, size = len(frames), frames.shape[-1]
    model = NCA().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, eps=1e-7)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, [2000], gamma=0.1)
    it, mode = 0, "train"
    if os.path.exists(SAVE_FILE):
        saved = torch.load(SAVE_FILE)
        model.load_state_dict(saved["model"])
        opt.load_state_dict(saved["opt"])
        sched.load_state_dict(saved["sched"])
        it, mode = saved["it"], "run"

    def save():
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "it": it}, SAVE_FILE)

    def loss_per_sample(xs, ages, offsets):
        target = frames[(frame_at(ages, n_frames) + offsets.clamp(min=0)) % n_frames]
        return ((xs[:, :4] - target) ** 2).mean((1, 2, 3))

    @torch.no_grad()
    def best_offset(xs, ages):
        base = frame_at(ages, n_frames)
        errs = torch.stack([((xs[:, :4] - frames[(base + o) % n_frames]) ** 2).mean((1, 2, 3))
                            for o in range(n_frames)], 1)
        return errs.argmin(1)

    @torch.no_grad()
    def make_report():
        # grow one sprite from a seed, keep (target, state, frame, step) at each check
        x, off, rows = model.seed(1, size), -1, []
        for t in range(1, GROW + n_frames * FRAME_STEPS + 1):
            x = model.step(x)
            a = torch.tensor([t], device=DEVICE)
            if bool(is_check(a)):
                if off < 0:
                    off = int(best_offset(x, a)[0])
                k = int((frame_at(a, n_frames)[0] + off) % n_frames)
                rows.append((frames[k], x[0].clone(), k, t))
        return rows

    tile = size * TILE_SCALE
    wsize = max(WATCH_SIZE, size)
    width = max(size * SCALE + wsize * WATCH_SCALE + 30, n_frames * (tile + 10) + 10)
    pygame.init()
    screen = pygame.display.set_mode(
        (width, max(size * SCALE, wsize * WATCH_SCALE, 2 * tile + 40) + 90))
    font = pygame.font.SysFont("consolas", 16)
    clock = pygame.time.Clock()

    def text(msg, xy, colour=(230, 230, 230)):
        screen.blit(font.render(msg, True, colour), xy)

    pool_x = model.seed(POOL_SIZE, size)
    pool_age = torch.zeros(POOL_SIZE, dtype=torch.long, device=DEVICE)
    pool_off = torch.full((POOL_SIZE,), -1, dtype=torch.long, device=DEVICE)   # -1 = not set yet
    pool_last = torch.zeros(POOL_SIZE, device=DEVICE)
    x, run_step, view_off = model.seed(1, wsize), 0, -1
    report, report_it, loss_value = None, 0, float("nan")
    radius = 6
    speed_i, step_credit = SPEEDS.index(1), 0.0
    plant_at = []
    yy, xx = torch.meshgrid(torch.arange(wsize, device=DEVICE), torch.arange(wsize, device=DEVICE),
                            indexing="ij")
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                if mode == "train" and it > 0:
                    save()
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN and event.key in (pygame.K_SPACE, pygame.K_r):
                if mode == "train" and it > 0:
                    save()
                mode, run_step, x, view_off = "run", 0, model.seed(1, wsize), -1
            if event.type == pygame.KEYDOWN and event.key == pygame.K_t and it < TRAIN_STEPS:
                mode = "train"
            if event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
                speed_i = min(speed_i + 1, len(SPEEDS) - 1)
            if event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN:
                speed_i = max(speed_i - 1, 0)
            if event.type == pygame.MOUSEWHEEL:
                radius = int(np.clip(radius + event.y, 1, 30))
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                plant_at.append(event.pos)

        model.wrap = WRAP_WHEN_WATCHING and not (mode == "train" and it < TRAIN_STEPS)
        if mode == "train" and it < TRAIN_STEPS:
            idx = torch.tensor(np.random.choice(POOL_SIZE, BATCH, replace=False), device=DEVICE)
            order = idx[pool_last[idx].argsort(descending=True)]      # worst first, replaced by a seed
            xb = torch.cat([model.seed(1, size), pool_x[order[1:]]])
            age = torch.cat([torch.zeros(1, dtype=torch.long, device=DEVICE), pool_age[order[1:]]])
            off = torch.cat([torch.full((1,), -1, dtype=torch.long, device=DEVICE), pool_off[order[1:]]])
            last = torch.cat([torch.zeros(1, device=DEVICE), pool_last[order[1:]]])
            loss, checks = 0.0, 0
            for _ in range(np.random.randint(*ROLLOUT)):
                xb = model.step(xb)
                age = age + 1
                c = is_check(age)
                if c.any():
                    new = c & (off < 0)
                    if new.any():
                        off = torch.where(new, best_offset(xb, age), off)
                    per = loss_per_sample(xb, age, off)
                    last = torch.where(c, per.detach(), last)
                    loss = loss + (per * c).sum()
                    checks += int(c.sum())
            pool_x[order], pool_age[order], pool_off[order], pool_last[order] = xb.detach(), age, off, last
            if checks == 0:
                continue
            loss = loss / checks
            opt.zero_grad()
            loss.backward()
            for p in model.parameters():
                p.grad /= p.grad.norm() + 1e-8
            opt.step()
            sched.step()
            it, loss_value = it + 1, loss.item()
            if report is None or it % REPORT_EVERY == 0:
                report, report_it = make_report(), it
            if it == TRAIN_STEPS:
                save()
                mode, run_step, x, view_off = "run", 0, model.seed(1, wsize), -1
            if it % 3:
                continue
            screen.fill((40, 40, 40))
            text(f"TRAINING   iteration {it}/{TRAIN_STEPS}   loss {loss_value:.5f}"
                 f"   (SPACE: stop and watch it walk)", (10, 8))
            text(f"Snapshot of the weights at iteration {report_it} (updates every {REPORT_EVERY}):"
                 f" one sprite grown from a seed, at each frame's check", (10, 30), (170, 170, 170))
            for i, (target, nca, k, t) in enumerate(report):
                x0 = 10 + i * (tile + 10)
                screen.blit(to_surface(target, TILE_SCALE), (x0, 55))
                screen.blit(to_surface(nca, TILE_SCALE), (x0, 55 + tile + 25))
                text(f"target frame {k + 1}", (x0, 55 + tile + 4), (170, 170, 170))
                text(f"NCA at step {t}", (x0, 55 + 2 * tile + 29), (170, 170, 170))
            pygame.display.flip()
        else:
            total = size * SCALE + 10 + wsize * WATCH_SCALE
            left = (width - total) // 2
            right = left + size * SCALE + 10

            def grid_cell(pos):
                gx, gy = (pos[0] - right) // WATCH_SCALE, (pos[1] - 40) // WATCH_SCALE
                return (gx, gy) if 0 <= gx < wsize and 0 <= gy < wsize else None

            mx, my = pygame.mouse.get_pos()
            under = grid_cell((mx, my))
            for pos in plant_at:
                cell = grid_cell(pos)
                if cell is not None:
                    x[0, :, cell[1], cell[0]] = 0.0
                    x[0, 3:, cell[1], cell[0]] = 1.0
            plant_at.clear()
            if pygame.mouse.get_pressed()[0] and under is not None:
                dx, dy = (xx - under[0]).abs(), (yy - under[1]).abs()
                if model.wrap:
                    dx, dy = torch.minimum(dx, wsize - dx), torch.minimum(dy, wsize - dy)
                x = x * ((dx ** 2 + dy ** 2) > radius ** 2).float()
            step_credit += SPEEDS[speed_i]
            with torch.no_grad():
                while step_credit >= 1:
                    step_credit -= 1
                    x = model.step(x)
                    run_step += 1
                    a = torch.tensor([run_step], device=DEVICE)
                    if view_off < 0 and bool(is_check(a)):
                        c0 = wsize // 2 - size // 2
                        view_off = int(best_offset(x[:, :, c0:c0 + size, c0:c0 + size], a)[0])
            a = torch.tensor([run_step], device=DEVICE)
            screen.fill((40, 40, 40))
            if view_off >= 0:
                k = int((frame_at(a, n_frames)[0] + view_off) % n_frames)
                screen.blit(to_surface(frames[k], SCALE), (left, 40))
                text(f"target: heading to frame {k + 1}", (left, 12))
            else:
                text(f"target: shown once grown (step {GROW + FRAME_STEPS - 1})", (left, 12))
            panel = wsize * WATCH_SCALE
            screen.blit(to_surface(x[0], WATCH_SCALE), (right, 40))
            text(f"NCA: step {run_step}   speed {SPEEDS[speed_i]:g}x", (right, 12))
            text(f"left click: erase   right click: new seed   wheel: eraser ({radius})"
                 f"   Up/Down: speed   R: restart   ESC: quit",
                 (left, max(size * SCALE, panel) + 50), (170, 170, 170))
            if under is not None:
                pygame.draw.circle(screen, (220, 60, 60), (mx, my), radius * WATCH_SCALE, 1)
            pygame.display.flip()
            clock.tick(60)


if __name__ == "__main__":
    main()
