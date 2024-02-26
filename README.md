# litmusAPI
This repository contains the LitmusAPI documentation developed using Swagger and Node-RED. Follow the instructions below to set up the project.

## Initial Setup 
To get started with the project, perform the following steps in the terminal or PowerShell:

1. **Navigate to the project directory**
```cd LitmusAPI```
<br>

2. **Clone the repository**
```git clone https://github.com/edwinmyp88/litmusAPI.git```
<br>

3. **Navigate to the project directory**
```cd litmusAPI``` 
* Use 'ls' (on Linux/MacOS) or 'dir' (on Windows) to view the project files

## Creating Your Own Branch 
To  work on a new feature or fix, create and switch to your own branch:

1. **Create a new branch with your name:**
    ```git branch sebastian```

2. **Switch to your branch:**
    ```git checkout sebastian```

## Checking Your Current Branch
To verify which branch you are currently on:
* Execute ```git branch```; the branch highlighted in green is your current branch.

## Commiting Your Changes
To commit changes to your branch:

1. **Stage your changes:**
```git add FILENAME``` or ```git add .``` to add all files

2. **Commit your changes with a message:**
```git commit -m 'YOUR COMMIT MESSAGE'```

3. **Check the status of your changes:**
```git status```

4. **Ensure your file is staged for commit(will appear in green).**

5. **Push your changes: (feature is the example branch name)**
```git push origin feature```

6. **Review your commit history:**
```git log```

## Syncing with the Main Branch
Before starting your work, ensure your branch is up to date with the main branch:

1. **Fetch the latest changes:**
```git fetch origin main```

2. **Merge the latest changes from the main branch into your current branch:**
```git pull origin main```

## Pushing Changes to Main
To integrate your changes into the main branch:

1. **Follow the usual commit process.**
2. **Prepare your branch for merging:**
* Fetch and pull the latest changes from origin, then merge with the main branch.
```
git fetch origin
git pull origin
git merge origin main
git push origin feature
```
3. **Create a Pull Request(PR) for your changes.**

## Handling Conflicts
In case of merge conflicts:
* Follow the fetching and pulling steps, then manually resolve the conflicts before pushing your changes and create a PR.

### For Conflicts in the Main Branch:
1. **Merge the feature branch into the main branch and resolve any conflicts:**
```git merge feature```
2. **Push the resolved changes to the main branch:**
```git push origin main```


