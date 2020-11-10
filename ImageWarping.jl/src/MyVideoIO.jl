__precompile__(true);

module MyVideoIO

using Images
using JSON

"""
    load(filename::String; stream::Int = 1)

Load a video from `filename` and return it as a 3D Array of RGB values.
"""
function load(filename::String; stream::Int = 1)
    # Get video info
    if !isfile(filename)
        error("The path does not define a file: '$(filename)'")
    end
    cmd = Cmd(`ffprobe -v quiet -print_format json -show_format -show_streams $(filename)`)
    j = read(cmd) |> String |> JSON.parse
    video_streams = filter(j["streams"]) do stream
        stream["codec_type"] == "video"
    end
    if length(video_streams) > 1
        warn("File has multiple video streams. Selecting stream $(stream).")
    end
    if length(video_streams) < 1
        error("File has no video streams.")
    end
    stream = video_streams[stream]
    h = stream["height"]
    w = stream["width"]
    # Not all containers will have the nb_frames info.
    # frames = parse(Int, stream["nb_frames"])
    # Read video data
    cmd = Cmd(`ffmpeg -v quiet -i $(filename) -f rawvideo -pix_fmt rgb24 - `)
    data = read(cmd)
    data = reinterpret(N0f8, data)
    frames = length(data) ÷ w ÷ h ÷ 3
    data = reshape(data, (3,w,h,frames))
    data = permutedims(data, (1,3,2,4))
    data = colorview(RGB, data)
end

save_presets = Dict(
    "lossless.mp4" => String["-c:v", "libx264", "-preset", "veryslow", "-qp", "0"],
)

function string_of_save_presets()
    mapreduce(kv -> string("- \"", kv[1], "\"", " => \"", join(kv[2], " "), "\"\n"), *, save_presets)
end

"""
    save(filename, data; ffmpeg_input_params = [], ffmpeg_output_params = [], fps::Int = 0, preset = "")

Save 3D Array of Color values `data` to `filename`, using ffmpeg. The available `presets` are:

$(string_of_save_presets())
"""
function save(
  filename :: String
, data :: Array{PixelType,3}
; ffmpeg_input_params  :: Vector{String} = String[]
, ffmpeg_output_params :: Vector{String} = String[]
, fps::Int = 0
, preset = ""
) where PixelType
    h,w,frames = size(data)
    append!(ffmpeg_output_params, get(save_presets, preset, []))
    if fps > 0
        fpscmd = ["-r", string(fps)]
        append!(ffmpeg_input_params , fpscmd)
        append!(ffmpeg_output_params, fpscmd)
    end
    if PixelType <: Color{T,1} where T || PixelType <: Real
        pixfmt = "gray"
        data = convert.(Gray{N0f8}, data)
        # append!(ffmpeg_output_params, ["-pix_fmt", pixfmt])
    elseif PixelType <: Color{T,3} where T
        pixfmt = "rgb24"
        data = convert.(RGB{N0f8}, data)
    end
    cmd = Cmd(`ffmpeg -y -v quiet -f rawvideo -pix_fmt $(pixfmt) -s $(w)x$(h) $(ffmpeg_input_params) -i - $(ffmpeg_output_params) $(filename)`)
    data = permutedims(data, (2,1,3))
    # Write video data
    open(cmd, "w") do io
        write(io,data)
    end
end

end
