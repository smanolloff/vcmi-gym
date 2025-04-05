cat <<-EOF >~/.gitconfig
[core]
    pager = less -R
    editor = vim
    excludesfile = ~/.gitignore_global
[color]
    ui = true
[url "https://github.com/"]
    insteadOf = git@github.com:
[alias]
    co = checkout
    ci = commit
    st = status
    br = branch
    d = diff
    h = log --pretty=format:\\"%C(magenta)%h%C(reset) %C(green)%ad%C(reset) %s%d %C(blue)[%an]%C(reset)\\" --graph --date=short -15
    hh = log --pretty=format:\\"%C(magenta)%h%C(reset) %C(green)%ad%C(reset) %s%d %C(blue)[%an]%C(reset)\\" --graph --date=short -15 --first-parent
EOF

cat <<-EOF >~/.gitignore_global
*.swp
*.log
*.orig
*.pyc
.venv
EOF
