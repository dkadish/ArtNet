const svgns = 'http://www.w3.org/2000/svg';
var image = "{{ image|safe }}";
var bboxes = {{ bboxes|safe }};
var probs = {{ probs|safe }};
var width = {{ width|safe }};
var height = {{ height|safe }};

// A little scaling function
function linearScale(num, in_min, in_max, out_min, out_max) {
    return (num - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

// Calculate opacity from threshold
function calcOpacity(v) {
    return linearScale(v, 0, 1, 0.2, 1);
}

// TODO: Would benefit from object with attribute names
function drawRects(container, data, threshold) {
    // Empty the container
    container.innerHTML = null;

    var rects = data.map(bb => {
        e = document.createElementNS(svgns, 'rect');
        e.setAttributeNS(null, 'x', bb[0]);
        e.setAttributeNS(null, 'y', bb[1]);
        e.setAttributeNS(null, 'width', bb[2] - bb[0]);
        e.setAttributeNS(null, 'height', bb[3] - bb[1]);
        e.setAttributeNS(null, 'opacity', calcOpacity(threshold));
        return e;
    });

    // Add them to container
    rects.forEach(rect => container.appendChild(rect));
}

function updateBoundingBoxes(container, bboxes, threshold) {
    // Create a mask of probable bboxes.
    var mask = probs.map(p => p >= threshold);
    var bboxes_probable = bboxes.filter((item, i) => mask[i]);
    console.log(bboxes_probable);
    document.querySelector("#so_many").innerText = bboxes_probable.length;
    document.querySelector("#prob-slider-echo").innerText = threshold;

    // And add them.
    drawRects(container, bboxes_probable, threshold);
}

// Place the bounding boxes into an existing SVG element.
var container = document.querySelector('#map');
var s = document.querySelector('svg#map_bboxes');
updateBoundingBoxes(s, bboxes, 0.6);

var slider = document.querySelector('#prob-slider')
slider.oninput = function(e) {
    updateBoundingBoxes(s, bboxes, this.value);
};

// Scale down the container which has both the original and the
// bounding boxes in it.
var scale = window.innerWidth / (width * 2);
var m = document.querySelector('#map');
m.style.transform = "scale(" + scale + ")";
m.style.height = height * scale + "px"
