-- Keep the deck's Julia excerpts tied to the build-executed feature atlas.
local source_path = "../../src/feature-atlas.md"

local function example(name)
  assert(name == "gaussian" or name == "population_pk",
    "unknown BRM deck example: " .. name)
  local file = assert(io.open(source_path, "r"))
  local source = file:read("*a")
  file:close()
  local marker = "\n" .. name .. " = (@brm begin"
  local first = assert(source:find(marker, 1, true), "missing example: " .. name)
  assert(not source:find(marker, first + 1, true), "ambiguous example: " .. name)
  local last = assert(source:find('"""', first, true), "missing example terminator")
  return source:sub(first + 1, last - 1):gsub("%s+$", "")
end

local function dedent(text)
  return text:gsub("^    ", ""):gsub("\n    ", "\n")
end

function CodeBlock(block)
  local name = block.attributes["brm-example"]
  if not name then return nil end
  assert(block.text:match("^%s*$"), "BRM source block must not contain a second code copy")
  local source = example(name)
  local part = block.attributes["brm-part"] or "full"
  if part ~= "full" then
    local separator = assert(source:find("\nend)((;", 1, true),
      "missing BRM declaration/data boundary")
    if part == "body" then
      local body_start = assert(source:find("\n", 1, true)) + 1
      source = dedent(source:sub(body_start, separator - 1))
    elseif part == "data" then
      local data_start = assert(source:find("\n", separator + 1, true)) + 1
      local data_end = assert(source:find("\n))", data_start, true))
      source = dedent(source:sub(data_start, data_end - 1))
    else
      error("unknown BRM excerpt part: " .. part)
    end
  end
  block.text = source
  block.attributes["brm-example"] = nil
  block.attributes["brm-part"] = nil
  return block
end
