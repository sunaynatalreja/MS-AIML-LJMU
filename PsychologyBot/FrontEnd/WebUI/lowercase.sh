#!/bin/bash
cd static/videos

for f in *.mp4; do
  lower=$(echo "$f" | tr '[:upper:]' '[:lower:]')
  if [[ "$f" != "$lower" ]]; then
    mv "$f" "$lower"
  fi
done

