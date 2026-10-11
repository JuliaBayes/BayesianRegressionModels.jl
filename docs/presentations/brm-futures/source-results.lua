-- Keep the deck's HSGP numbers tied to the committed study results.
-- Values are read at render time; an unknown arm, field or file fails the render.
local refresh_dir = "../../../research/centering_refresh/results/hsgp/"
local source_dir = "../../../research/adaptive_centering/results/source-faithful/"

local function read_tsv(path)
  local file = assert(io.open(path, "r"), "missing BRM deck result file: " .. path)
  local header, rows = nil, {}
  for line in file:lines() do
    line = line:gsub("\r$", "") -- the refresh harness writes CRLF rows
    local cells = {}
    for cell in (line .. "\t"):gmatch("([^\t]*)\t") do cells[#cells + 1] = cell end
    if header == nil then
      header = cells
    else
      assert(#cells == #header, "ragged row in BRM deck result file: " .. path)
      local row = {}
      for i, name in ipairs(header) do row[name] = cells[i] end
      rows[#rows + 1] = row
    end
  end
  file:close()
  assert(header and #rows > 0, "empty BRM deck result file: " .. path)
  return rows
end

local function number(text, what)
  local value = tonumber(text)
  assert(value ~= nil, "non-numeric " .. what .. ": " .. tostring(text))
  return value
end

-- Rounded integer with thousands separators: 4648422.4 -> 4,648,422.
local function grouped(x)
  local digits = string.format("%d", math.floor(x + 0.5))
  return (digits:reverse():gsub("(%d%d%d)", "%1,"):reverse():gsub("^,", ""))
end

-- Relative efficiency, three significant figures, as on the case-study page.
local function ratio(x)
  return string.format("%.3g", x) .. "×"
end

-- An error maximum rounded UP to three significant figures, so it stays a bound.
local function error_bound(x)
  assert(x > 0, "error bound must be positive")
  local mantissa, exponent = string.format("%.2e", x):match("^(%d%.%d%d)e([-+]%d+)$")
  mantissa, exponent = tonumber(mantissa), tonumber(exponent)
  if mantissa * 10.0 ^ exponent < x then mantissa = mantissa + 0.01 end
  if mantissa >= 9.995 then mantissa, exponent = mantissa / 10, exponent + 1 end
  return string.format("%.2fe%d", mantissa, exponent)
end

local function only(rows, matches, what)
  local found
  for _, row in ipairs(rows) do
    if matches(row) then
      assert(found == nil, "ambiguous " .. what)
      found = row
    end
  end
  assert(found ~= nil, "missing " .. what)
  return found
end

local hsgp_fields = {
  method = function(row) return row.method end,
  draws = function(row) return grouped(number(row.draws, "draws")) end,
  qois = function(row) return grouped(number(row.qois, "quantity count")) end,
  divergences = function(row) return grouped(number(row.divergences, "divergences")) end,
  rhat = function(row) return string.format("%.4f", number(row.max_split_rhat, "split R-hat")) end,
  bulk = function(row) return grouped(number(row.min_bulk_ess, "bulk ESS")) end,
  tail = function(row) return grouped(number(row.min_tail_ess, "tail ESS")) end,
  gradients = function(row) return grouped(number(row.total_gradients, "total gradients")) end,
  sampling = function(row) return ratio(number(row.relative_sampling_efficiency, "sampling efficiency")) end,
  total = function(row) return ratio(number(row.relative_total_efficiency, "total efficiency")) end,
}

local function hsgp(args)
  local key = assert(args[1], "brm-hsgp needs an arm or a record name")
  if key == "frame-checks" then
    return grouped(#read_tsv(refresh_dir .. "frame_checks.tsv"))
  elseif key == "warmuphmc" then
    local arm = assert(args[2], "brm-hsgp warmuphmc needs an arm")
    local package = only(read_tsv(refresh_dir .. arm .. "/packages.tsv"),
      function(row) return row.package == "WarmupHMC" end, "WarmupHMC package for arm " .. arm)
    assert(package.git_sha:match("^%x+$"), "WarmupHMC for arm " .. arm .. " has no git revision")
    return package.git_sha:sub(1, 7)
  elseif key == "posthoc-source-matches" then
    -- The deck states that the refreshed post-hoc position fit uses the source
    -- reproduction's selections; count them only if every one is identical.
    local selected = {}
    for _, row in ipairs(read_tsv(refresh_dir .. "controls.tsv")) do
      if row.arm == "Post-hoc position" then
        selected[row.group .. "/" .. row.id] = number(row.centeredness, "refreshed centeredness")
      end
    end
    local count = 0
    for _, row in ipairs(read_tsv(source_dir .. "centeredness.tsv")) do
      for column, group in pairs({ mean = "Mean GP", log_scale = "Log-SD GP" }) do
        local key = group .. "/" .. row.basis
        local source = number(row[column], "source centeredness")
        assert(selected[key] == source, "refreshed post-hoc choice differs from source at " .. key)
        selected[key] = nil
        count = count + 1
      end
    end
    assert(next(selected) == nil, "refreshed post-hoc controls include choices absent from the source")
    return grouped(count)
  end
  local field = assert(hsgp_fields[args[2] or ""],
    "unknown brm-hsgp field: " .. tostring(args[2]))
  local row = only(read_tsv(refresh_dir .. "comparison.tsv"),
    function(row) return row.case == "hsgp" and row.arm == key end, "HSGP arm " .. key)
  return field(row)
end

local function source_audit(args)
  local rows = read_tsv(source_dir .. "partial_source_density_gradient_audit.tsv")
  local function maximum(column)
    local largest = 0.0
    for _, row in ipairs(rows) do
      largest = math.max(largest, number(row[column], column))
    end
    return largest
  end
  local fields = {
    positions = function() return grouped(#rows) end,
    max_density = function() return error_bound(maximum("density_absolute_error")) end,
    max_gradient = function() return error_bound(maximum("max_gradient_absolute_error")) end,
  }
  local field = assert(fields[args[1] or ""], "unknown brm-source-audit field: " .. tostring(args[1]))
  return field()
end

local function shortcode(handler)
  return function(args)
    local words = {}
    for i, arg in ipairs(args) do words[i] = pandoc.utils.stringify(arg) end
    return pandoc.Str(handler(words))
  end
end

return {
  ["brm-hsgp"] = shortcode(hsgp),
  ["brm-source-audit"] = shortcode(source_audit),
}
