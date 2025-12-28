# Contributing Guide

This repository is a fork maintained by **Maxim Strekolovskiy**, based on **Second Me** by **Mindverse** (Apache-2.0).

If you want to contribute to **this fork**, please open issues/PRs in this repository.
If you want to contribute to the **upstream** project, use the upstream repository.

The Second Me community welcomes various forms of contributions, including code, non-code contributions, documentation, and more.

## How to Contribute

| Contribution Type | Details |
|------------------|---------|
| Report a bug | You can file an issue to report a bug with Second Me |
| Contribute code | You can contribute your code by fixing a bug or implementing a feature |
| Code Review | If you are an active contributor or committer of Second Me, you can help us review pull requests |
| Documentation | You can contribute documentation changes by fixing a documentation bug or proposing new content |

## Before Contributing
* Upstream contributions: Sign [CLA of Mindverse](https://cla-assistant.io/mindverse/Second-Me)
* This fork: no CLA required unless explicitly requested by the maintainer.
  
## Here is a checklist to prepare and submit your PR (pull request).
* Create your own GitHub branch by forking Second Me
* Checkout [README](README.md) for how to deploy Second Me.
* Push changes to your personal fork.
* Create a PR with a detailed description, if commit messages do not express themselves.
* Submit PR for review and address all feedbacks.
* Wait for merging (done by committers).

## Branch Management Strategy

We follow a structured branching strategy to manage releases and contributions from both internal and external contributors.

### Branch Structure

```
master (stable version)
    ^
    |
release/vX.Y.Z (release preparation branch)
    ^
    |
develop (development integration branch)
    ^
    |
feature/* (feature branches) / hotfix/* (hotfix branches)
```

## Development Workflow

```
                 hotfix/fix-bug
                /       \
master ---------+---------+-----+--- ... --> Stable Version
                \               /
                 \             /
release/v1.0 -----+-----------+--- ... --> Release Version
                     \         /
                      \       /
develop --------------+-----+---+--- ... --> Development Integration
                     /       /
                    /       /
feature/new-feature +----------- ... --> Feature Development (from master)
```

### Step 1: Fork and Clone (External Contributors Only)
If you're an external contributor, you need to fork the repository first:

1. Visit https://github.com/Mindverse/Second-Me
2. Click the "Fork" button in the top-right corner
3. Clone your fork to your local machine:
```bash
cd working_dir
# Replace USERNAME with your GitHub username
git clone git@github.com:USERNAME/Second-Me.git
cd Second-Me
```

4. Configure upstream remote:
```bash
# Add the upstream repository
git remote add upstream git@github.com:Mindverse/Second-Me.git

# Verify your remotes
git remote -v
```

### Step 2: Create a Feature Branch
All contributors should create feature branches from the `master` branch:

```bash
# First, ensure you have the latest changes
git fetch origin  # or upstream if you're working with a fork

# Checkout the master branch
git checkout master

git pull

# Create your feature branch from master
git checkout -b feature/your-feature-name
```

### Step 3: Develop Your Feature
- Make changes in your feature branch
- Commit regularly with descriptive messages
- Follow the project's coding style
- Add tests if applicable
- Update documentation as needed

### Step 4: Commit Your Changes
```bash
# Add your changes
git add <filename>
# Or git add -A for all changes

# Commit with a clear message
git commit -m "feat: add new feature X"
```

### Step 5: Update Your Branch Before Submitting
Before submitting your PR, update your feature branch with the latest changes:

```bash
# Fetch latest changes
git fetch origin  # or upstream if you're working with a fork

# Rebase your feature branch
git checkout feature/your-feature-name
git rebase origin/master  # or upstream/master for forked repos
```

If you're an external contributor, you may need to push to your fork:
```bash
git push origin feature/your-feature-name
```

### Step 6: Create a Pull Request
1. Visit the repository (or your fork)
2. Click "Compare & Pull Request"
3. Select:
   - Base repository: `Mindverse/Second-Me`
   - Base branch: `develop` (all features and fixes go to develop first)
   - Head repository: Your repository
   - Compare branch: `feature/your-feature-name`
4. Fill in the PR template with:
   - Clear description of your changes
   - Any related issues
   - Testing steps if applicable
   - Target version if applicable

### Step 7: Address Review Feedback
- Maintainers will review your PR
- Address any feedback by making requested changes
- Push new commits to your feature branch
- Your PR will be updated automatically

### Step 8: PR Approval and Merge
- All checks must pass before merging
- Once approved, maintainers will merge your PR to the appropriate branch
- Your contribution will be included in the next release cycle

## Release Management

The following describes how releases are managed by project maintainers:

```
                               PR Merge Flow
                               |
master -------------------------+------ ... --> Stable Version
                               |
                               |
release/vX.Y.Z ---------------+------- ... --> Release Version
                             / |
                            /  |
develop --------------------+--+------ ... --> Development Integration
     ^                         
     |                          
feature branches --------------+
```

### Creating a Release
1. When `develop` branch contains all features planned for a release, a `release/vX.Y.Z` branch is created
2. Only bug fixes and release preparation commits are added to the release branch
3. After thorough testing, the release branch is merged to `master`
4. The release is tagged in `master` with the version number

### PR Merge Strategy
- All feature PRs are initially merged to the `develop` branch
- Critical bug fixes may be merged directly to the current `release` branch
- Maintainers are responsible for ensuring PRs are merged to the appropriate branch

### Hotfixes
1. For critical bugs in production, create a `hotfix/fix-description` branch from `master`
2. Fix the issue and create a PR targeting `master`
3. After approval, merge to both `master` and `develop` (and current `release` branch if exists)

## Tips for Successful Contributions
- Create focused, single-purpose PRs
- Follow the project's code style and conventions
- Write clear commit messages
- Keep your fork updated to avoid merge conflicts
- Be responsive during the review process
- Ask questions if anything is unclear
