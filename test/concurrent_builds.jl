# All tasks reach the barrier before any build starts. Blocking on the Event
# yields, so this also works when there are more jobs than worker threads.
function concurrent_builds(f, inputs)
    ready = Channel{Nothing}(length(inputs))
    release = Base.Event()
    tasks = [Threads.@spawn begin
        put!(ready, nothing)
        wait(release)
        f(input)
    end for input in inputs]
    for _ in tasks
        take!(ready)
    end
    notify(release)
    fetch.(tasks)
end
