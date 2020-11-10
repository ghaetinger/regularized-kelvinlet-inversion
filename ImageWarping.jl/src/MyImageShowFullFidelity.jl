module MyImageShowFullFidelity

using FileIO
using Colors

import Base.show
function Base.show(io::IO, mime::MIME"image/png", img::AbstractMatrix{C}) where C<:Colorant
    if Base.has_offset_axes(img)
        img = collect(img)
    end
    FileIO.save(FileIO.Stream(format"PNG", io), img)
end

end
