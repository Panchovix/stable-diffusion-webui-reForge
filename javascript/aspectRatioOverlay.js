let arFrameTimeout;
let e_img2img;
let e_inpaint;
let arPreviewRect;

function dimensionChange(width_input, height_input) {
    var inImg2img = gradioApp().querySelector("#tab_img2img").style.display == "block";
    if (!inImg2img) {
        return;
    }
    method = gradioApp().querySelector('#img2img_inpaint_full_res > div > div > div > div > input').value;
    if (method != 'Whole picture') {
        return;
    }

    let targetWidth = width_input.value * 1.0;
    let targetHeight = height_input.value * 1.0;

    var targetElement = null;

    var tabIndex = get_tab_index('mode_img2img');

    if (tabIndex == 0) { // img2img
        if (!e_img2img) {
            if (opts.forge_canvas_plain) {
                e_img2img = gradioApp().querySelector('#img2img_image div[class=forge-image-container-plain] img');
            }
            else {
                e_img2img = gradioApp().querySelector('#img2img_image div[class=forge-image-container] img');
            }
        }
        targetElement = e_img2img;
    }
    else if (tabIndex == 1) { // Inpaint upload
        if (!e_inpaint) {
            e_inpaint = gradioApp().querySelector('#img_inpaint_base div[data-testid=image] img');
        }
        targetElement = e_inpaint;
    }

    if (targetElement) {
        var viewportOffset = targetElement.getBoundingClientRect();
        var viewportscale = Math.min(targetElement.clientWidth / targetElement.naturalWidth, targetElement.clientHeight / targetElement.naturalHeight);

        var scaledx = targetElement.naturalWidth * viewportscale;
        var scaledy = targetElement.naturalHeight * viewportscale;

        var clientRectTop = (viewportOffset.top + window.scrollY);
        var clientRectLeft = (viewportOffset.left + window.scrollX);
        var clientRectCentreY = clientRectTop + (targetElement.clientHeight / 2);
        var clientRectCentreX = clientRectLeft + (targetElement.clientWidth / 2);

        var arscale = Math.min(scaledx / targetWidth, scaledy / targetHeight);
        var arscaledx = targetWidth * arscale;
        var arscaledy = targetHeight * arscale;

        var arRectTop = clientRectCentreY - (arscaledy / 2);
        var arRectLeft = clientRectCentreX - (arscaledx / 2);
        var arRectWidth = arscaledx;
        var arRectHeight = arscaledy;

        arPreviewRect.style.top = arRectTop + 'px';
        arPreviewRect.style.left = arRectLeft + 'px';
        arPreviewRect.style.width = arRectWidth + 'px';
        arPreviewRect.style.height = arRectHeight + 'px';

        clearTimeout(arFrameTimeout);
        arFrameTimeout = setTimeout(function() {
            arPreviewRect.style.display = 'none';
        }, 2000);

        arPreviewRect.style.display = 'block';
    }
}

onUiLoaded(function() {
    arPreviewRect = gradioApp().querySelector('#imageARPreview');
    if (arPreviewRect) {
        arPreviewRect.style.display = 'none';
    }
    else {
        arPreviewRect = document.createElement('div');
        arPreviewRect.id = "imageARPreview";
        gradioApp().appendChild(arPreviewRect);
    }

    width_input  = gradioApp().querySelectorAll('#img2img_width input');
    height_input = gradioApp().querySelectorAll('#img2img_height input');

    width_input.forEach(function(e) {
        e.addEventListener('input', function(e) {
            dimensionChange(width_input[0], height_input[0]);
        });
    });
    height_input.forEach(function(e) {
        e.addEventListener('input', function(e) {
            dimensionChange(width_input[0], height_input[0]);
        });
    });
});
