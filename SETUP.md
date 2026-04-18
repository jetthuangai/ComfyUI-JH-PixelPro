# Setup guide — ComfyUI-JH-PixelPro

Guide for maintainers setting the pack up for development for the first time.

> If you just want to **install** the pack (consumer), the README.md is enough. This file is for maintainers.

## 1. Initialise a standalone git repo for the pack

The pack pushes to GitHub as an independent repository, unrelated to the surrounding ComfyUI repo.

> ⚠️ **Sandbox note:** if a broken `.git/` directory is present (e.g. from a half-completed scaffold), delete it before running `git init` again.

Run from a real terminal (PowerShell / bash on your machine):

```bash
cd ComfyUI/custom_nodes/ComfyUI-JH-PixelPro

# Remove a broken .git/ if one exists
rm -rf .git          # macOS/Linux
# or
rmdir /s /q .git     # Windows PowerShell

# First-time init
git init -b main
git add .
git commit -m "chore: scaffold ComfyUI-JH-PixelPro (Phase 1)"

# Point to the GitHub remote (create the empty repo there first)
git remote add origin git@github.com:<user>/ComfyUI-JH-PixelPro.git
git push -u origin main
```

## 2. Verify the pack is not tracked by the surrounding ComfyUI repo

```bash
cd ComfyUI
git check-ignore -v custom_nodes/ComfyUI-JH-PixelPro
# Expected: .gitignore:<line>:/custom_nodes/    custom_nodes/ComfyUI-JH-PixelPro
```

If the output is empty the pack will be swallowed by the ComfyUI repo — fix `ComfyUI/.gitignore` immediately.

## 3. Install dev dependencies

```bash
cd ComfyUI/custom_nodes/ComfyUI-JH-PixelPro
pip install -r requirements-dev.txt
```

## 4. Smoke test after install

```bash
# Confirm the pack loads inside ComfyUI
# (start ComfyUI, open the web UI, and check that the image/pixelpro/ menu appears)

# Pytest — collection must run cleanly without import errors.
pytest --collect-only
```

## 5. Git boundary checklist (per commit)

- [ ] `git rev-parse --show-toplevel` inside the pack must resolve to `ComfyUI-JH-PixelPro/`, **not** `ComfyUI/`.
- [ ] `git status` lists no files outside the pack.
- [ ] `git remote -v` points only at the pack's GitHub repo, not at any upstream ComfyUI remote.
