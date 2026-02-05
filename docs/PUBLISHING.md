# Publishing to PyPI

This guide explains how to set up and publish TempoEval to PyPI.

## 🔐 Setting Up Trusted Publishing (Recommended)

PyPI now supports "Trusted Publishing" which uses OpenID Connect to authenticate GitHub Actions without storing API tokens.

### Step 1: Create a PyPI Account

1. Go to [https://pypi.org/account/register/](https://pypi.org/account/register/)
2. Create an account and verify your email

### Step 2: Configure Trusted Publisher on PyPI

1. Go to [https://pypi.org/manage/account/publishing/](https://pypi.org/manage/account/publishing/)
2. Click "Add a new pending publisher"
3. Fill in:
   - **PyPI Project Name**: `tempoeval`
   - **Owner**: `DataScienceUIBK`
   - **Repository name**: `tempoeval`
   - **Workflow name**: `publish.yml`
   - **Environment name**: `pypi` (for production) or `testpypi` (for testing)
4. Click "Add"

### Step 3: Configure GitHub Repository

1. Go to your GitHub repo → Settings → Environments
2. Create environment `pypi`:
   - Add protection rules if desired (e.g., require approval)
3. Create environment `testpypi`:
   - This is for testing releases first

### Step 4: Create a Release

1. Go to GitHub → Releases → "Create a new release"
2. Create a new tag (e.g., `v0.1.0`)
3. Fill in release notes
4. Click "Publish release"

The workflow will automatically:
1. Build the package
2. Publish to TestPyPI first
3. Then publish to PyPI

## 🧪 Testing Locally

Before publishing, test the build locally:

```bash
# Install build tools
pip install build twine

# Build the package
python -m build

# Check the package
twine check dist/*

# Test upload to TestPyPI (optional)
twine upload --repository testpypi dist/*
```

## 📦 Manual Publishing (Alternative)

If you prefer manual publishing:

```bash
# Install tools
pip install build twine

# Build
python -m build

# Upload to PyPI
twine upload dist/*
```

You'll need to create an API token at [https://pypi.org/manage/account/token/](https://pypi.org/manage/account/token/)

## ✅ Post-Publishing Checklist

- [ ] Package appears on PyPI: https://pypi.org/project/tempoeval/
- [ ] Install works: `pip install tempoeval`
- [ ] Basic import works: `python -c "import tempoeval; print(tempoeval.__version__)"`
- [ ] Documentation badge shows correct version
- [ ] GitHub release notes are complete
