# Secure Secrets Management Setup

This project uses Claude Code's **SessionStart hook** to securely load API keys and credentials without committing them to the repository.

## How It Works

1. **SessionStart Hook** (`~/.claude/hooks/session-start.sh`) runs automatically when your session starts
2. It loads secrets from `~/.claude/secrets.env` into the session environment
3. Secrets persist throughout the entire Claude Code session via `CLAUDE_ENV_FILE`
4. **No secrets are committed to the repository** - they stay in your user profile

## Setup Instructions

### 1. Create Your Secrets File

Copy the example and add your real API keys:

```bash
cp ~/.claude/secrets.env.example ~/.claude/secrets.env
chmod 600 ~/.claude/secrets.env
```

Edit `~/.claude/secrets.env` and add your credentials:

```bash
# OpenAI API Key (for openai-clojure dependency)
export OPENAI_API_KEY="sk-..."

# Other API keys as needed
export DATABASE_URL="postgresql://user:pass@host:5432/dbname"
export HASS_TOKEN="your-home-assistant-token"
export HASS_URL="http://homeassistant.local:8123"
```

### 2. Verify the Hook is Registered

The SessionStart hook is already configured in `~/.claude/settings.json`:

```json
{
  "hooks": {
    "SessionStart": [
      {
        "matcher": "",
        "hooks": [{
          "type": "command",
          "command": "~/.claude/hooks/session-start.sh"
        }]
      }
    ]
  }
}
```

### 3. Test the Setup

After creating `~/.claude/secrets.env`, the next time you start a Claude Code session:

```bash
# In any bash command, your secrets will be available
echo $OPENAI_API_KEY
# Should show: sk-...

# In Clojure REPL
lein repl
(System/getenv "OPENAI_API_KEY")
# Should show your API key
```

## Important Security Notes

✅ **DO:**
- Store secrets in `~/.claude/secrets.env` (NOT in the repository)
- Use restrictive permissions: `chmod 600 ~/.claude/secrets.env`
- Keep the `.gitignore` updated to exclude all secret files
- Use the example file for documentation, never real secrets

❌ **DON'T:**
- Never commit `secrets.env` to git
- Don't put API keys directly in `settings.json` if it's committed
- Don't use `export` in regular bash commands (won't persist)
- Don't share your `secrets.env` file

## Files Structure

```
~/.claude/
  ├── secrets.env.example    # Template (safe to share)
  ├── secrets.env            # YOUR REAL SECRETS (gitignored, user-only)
  ├── hooks/
  │   └── session-start.sh   # Hook that loads secrets
  └── settings.json          # Hook registration

project/
  ├── .gitignore            # Updated to exclude secrets
  └── SECRETS_SETUP.md      # This file
```

## What's Protected

The `.gitignore` now excludes:
- `.env`, `.env.local`, `.env.*.local`
- `secrets.env`
- `credentials.json`, `api-keys.json`
- `*.pem`, `*.key`
- `.claude/secrets.env`

## Verify Nothing is Committed

```bash
# Check if any secrets are in git history
git log --all --full-history -- "*secrets*" "*credentials*" "*.env"

# Verify .gitignore is working
git check-ignore -v .env secrets.env .claude/secrets.env
```

## For Team Collaboration

**Share safely:**
- ✅ Commit `.gitignore` updates
- ✅ Commit `secrets.env.example` with placeholders
- ✅ Commit this documentation
- ✅ Share the SessionStart hook setup

**Never share:**
- ❌ Your actual `secrets.env` file
- ❌ Any files with real API keys
- ❌ `.env` files with real credentials

## Troubleshooting

**Secrets not loading?**
1. Check file exists: `ls -la ~/.claude/secrets.env`
2. Check permissions: `ls -l ~/.claude/hooks/session-start.sh` (should be executable)
3. Check syntax: `bash -n ~/.claude/hooks/session-start.sh`
4. Test manually: `~/.claude/hooks/session-start.sh`

**Hook not running?**
1. Verify registration: `cat ~/.claude/settings.json | grep SessionStart`
2. Check hook output in Claude Code session logs
3. Make sure hook exits with code 0 (success)

## Alternative: Project-Specific .env Files

For local development only (not recommended for web sessions):

```bash
# Create project .env (gitignored)
cat > .env << 'EOF'
OPENAI_API_KEY=sk-...
DATABASE_URL=postgresql://...
EOF

# Load in your shell
source .env

# Or use with lein
lein run
```

**Note:** This approach doesn't work well with Claude Code web sessions because environment variables don't persist between commands. The SessionStart hook approach is preferred.

## Resources

- [Claude Code Hooks Documentation](https://docs.claude.ai/docs/claude-code)
- [Environment Variables Best Practices](https://12factor.net/config)
- [Git Secrets Prevention](https://git-scm.com/book/en/v2/Git-Tools-Credential-Storage)
