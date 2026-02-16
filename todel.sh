curl --request POST \
  --url https://gateway.phyagi.net/api/chat/completions \
  --header 'Authorization: Bearer d4ef0743379f4fc992f9d02aba091743' \
  --header 'Content-Type: application/json' \
  --data '{
    "model": "gpt-4o",
    "tier": "base",
    "messages": [
      {"role": "user", "content": "Draft three alternate subject lines for a launch announcement."}
    ],
    "temperature": 0.4
  }'
 