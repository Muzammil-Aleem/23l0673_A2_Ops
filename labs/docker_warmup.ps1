# Task 6.1 — Docker warm-up with a ready-made image (nginx). Run each block, screenshot it.

# 1. run a container in the background and map host port 8080 -> container port 80
docker run -d --name warmup-nginx -p 8080:80 nginx:1.27-alpine
docker ps                                        # list running containers
# open http://localhost:8080 in a browser (screenshot)

# 2. mount a volume: serve our own page from a host folder (bind mount) ...
New-Item -ItemType Directory -Force "$PWD\labs\_www" | Out-Null
"<h1>Hello from a mounted volume - $env:USERNAME $(Get-Date)</h1>" | Out-File -Encoding utf8 "$PWD\labs\_www\index.html"
docker rm -f warmup-nginx
docker run -d --name warmup-nginx -p 8080:80 -v "${PWD}\labs\_www:/usr/share/nginx/html:ro" nginx:1.27-alpine
# ... and a named volume managed by Docker
docker volume create warmup-data
docker run --rm -v warmup-data:/data alpine sh -c "echo persisted > /data/note.txt"
docker run --rm -v warmup-data:/data alpine cat /data/note.txt   # data outlived the first container

# 3. read logs (refresh the browser a few times first)
docker logs --tail 20 warmup-nginx

# 4. open a shell inside it
docker exec -it warmup-nginx sh
#   inside:  whoami; ls /usr/share/nginx/html; cat /etc/os-release; exit

# 5. list and remove containers and images
docker ps -a
docker stop warmup-nginx
docker rm warmup-nginx
docker images
docker rmi nginx:1.27-alpine alpine
docker volume rm warmup-data
docker ps -a; docker images
