// A full size 'lightbox' preview modal shown when left clicking on gallery previews
function closeModal() {
    lightBoxModal.style.display = "none";
}

function showModal(event) {
    const source = event.target || event.srcElement;
    lightBoxToggle.innerHTML = opts.js_live_preview_in_modal_lightbox ? "&#x1F5C7;" : "&#x1F5C6;";

    lightBoxImage.src = source.src;
    if (lightBoxImage.style.display === 'none') {
        lightBoxModal.style.setProperty('background-image', 'url(' + source.src + ')');
    }
    lightBoxModal.style.display = "flex";
    lightBoxModal.focus();

    event.stopPropagation();
}

function negmod(n, m) {
    return ((n % m) + m) % m;
}

function updateOnBackgroundChange() {
    if (lightBoxImage && lightBoxImage.offsetParent) {
        let currentButton = selected_gallery_button();
        let preview = gradioApp().querySelectorAll('.livePreview > img');
        if (opts.js_live_preview_in_modal_lightbox && preview.length > 0) {
            // show preview image if available
            lightBoxImage.src = preview[preview.length - 1].src;
        } else if (currentButton?.children?.length > 0 && lightBoxImage.src != currentButton.children[0].src) {
            lightBoxImage.src = currentButton.children[0].src;
            if (lightBoxImage.style.display === 'none') {
                lightBoxModal.style.setProperty('background-image', `url(${lightBoxImage.src})`);
            }
        }
    }
}

function modalImageSwitch(offset) {
    var galleryButtons = all_gallery_buttons();

    if (galleryButtons.length > 1) {
        var result = selected_gallery_index();

        if (result != -1) {
            var nextButton = galleryButtons[negmod((result + offset), galleryButtons.length)];
            nextButton.click();
            lightBoxImage.src = nextButton.children[0].src;
            if (lightBoxImage.style.display === 'none') {
                lightBoxModal.style.setProperty('background-image', `url(${lightBoxImage.src})`);
            }
            setTimeout(function() {
                lightBoxModal.focus();
            }, 10);
        }
    }
}

function modalNextImage(event) {
    modalImageSwitch(1);
    event.stopPropagation();
}

function modalPrevImage(event) {
    modalImageSwitch(-1);
    event.stopPropagation();
}

function modalKeyHandler(event) {
    switch (event.key) {
    case "ArrowLeft":
        modalPrevImage(event);
        break;
    case "ArrowRight":
        modalNextImage(event);
        break;
    case "Escape":
        closeModal();
        break;
    case "u":
	    if (get_uiCurrentTab().innerText == 'Txt2img') {
            gradioApp().getElementById('txt2img_upscale').click();
        }
        break;
    case "z":
        modalZoomToggle();
        break;
    case "t":
        modalTileImageToggle();
        break;
    }
}

function setupImageForLightbox(e) {
    if (e.dataset.modded) {
        return;
    }

    e.dataset.modded = true;
    e.style.cursor = 'pointer';
    e.style.userSelect = 'none';

    e.addEventListener('mousedown', function(evt) {
        if (evt.button == 1) {
            open(evt.target.src);
            evt.preventDefault();
            return;
        }
    }, true);

    e.addEventListener('click', function(evt) {
        if (!opts.js_modal_lightbox || evt.button != 0) return;

        modalZoomSet(opts.js_modal_lightbox_initially_zoomed);
        evt.preventDefault();
        showModal(evt);
    }, true);

}

function modalZoomSet(enable) {
    lightBoxImage.classList.toggle('modalImageFullscreen', !!enable);
}

function modalZoomToggle(event) {
    modalZoomSet(!lightBoxImage.classList.contains('modalImageFullscreen'));
    event.stopPropagation();
}

function modalLivePreviewToggle(event) {
    opts.js_live_preview_in_modal_lightbox = !opts.js_live_preview_in_modal_lightbox;
    lightBoxToggle.innerHTML = opts.js_live_preview_in_modal_lightbox ? "&#x1F5C7;" : "&#x1F5C6;";
    event.stopPropagation();
}

function modalTileImageToggle(event) {
    const isTiling = lightBoxImage.style.display === 'none';
    if (isTiling) {
        lightBoxImage.style.display = 'block';
        lightBoxModal.style.setProperty('background-image', 'none');
    } else {
        lightBoxImage.style.display = 'none';
        lightBoxModal.style.setProperty('background-image', `url(${lightBoxImage.src})`);
    }

    event.stopPropagation();
}

// how to get this out of onUiUpdate into something sensible?
// run after generation returns?
// limit to currently active tab?

/*
function setupgallery() {
	alert("sg");
    var fullImg_preview = gradioApp().querySelectorAll('.gradio-gallery > button > button > img, .gradio-gallery > .livePreview');
    if (fullImg_preview != null) {
        fullImg_preview.forEach(setupImageForLightbox);
    }
    updateOnBackgroundChange();
	
}
*/

onUiUpdate(function() {
    var fullImg_preview = gradioApp().querySelectorAll('.gradio-gallery > button > button > img, .gradio-gallery > .livePreview');
    if (fullImg_preview != null) {
        fullImg_preview.forEach(setupImageForLightbox);
    }
    updateOnBackgroundChange();
});

let lightBoxModal = undefined;
let lightBoxImage = undefined;
let lightBoxToggle = undefined;

document.addEventListener("DOMContentLoaded", function() {
    lightBoxModal = document.createElement('div');
    lightBoxModal.onclick = closeModal;
    lightBoxModal.id = "lightboxModal";
    lightBoxModal.tabIndex = 0;
    lightBoxModal.addEventListener('keydown', modalKeyHandler, true);

    const modalControls = document.createElement('div');
    modalControls.className = 'modalControls gradio-container';
    lightBoxModal.append(modalControls);

    const modalZoom = document.createElement('span');
    modalZoom.className = 'modalZoom cursor';
    modalZoom.innerHTML = '&#10529;';
    modalZoom.addEventListener('click', modalZoomToggle, true);
    modalZoom.title = "Toggle zoomed view";
    modalControls.appendChild(modalZoom);

    const modalTileImage = document.createElement('span');
    modalTileImage.className = 'modalTileImage cursor';
    modalTileImage.innerHTML = '&#8862;';
    modalTileImage.addEventListener('click', modalTileImageToggle, true);
    modalTileImage.title = "Preview tiling";
    modalControls.appendChild(modalTileImage);

    lightBoxToggle = document.createElement('span');
    lightBoxToggle.className = 'modalToggleLivePreview cursor';
    lightBoxToggle.id = "modal_toggle_live_preview";
    lightBoxToggle.innerHTML = "&#x1F5C6;";
    lightBoxToggle.onclick = modalLivePreviewToggle;
    lightBoxToggle.title = "Toggle live preview";
    modalControls.appendChild(lightBoxToggle);

    const modalClose = document.createElement('span');
    modalClose.className = 'modalClose cursor';
    modalClose.innerHTML = '&times;';
    modalClose.onclick = closeModal;
    modalClose.title = "Close image viewer";
    modalControls.appendChild(modalClose);

    lightBoxImage = document.createElement('img');
    lightBoxImage.id = 'modalImage';
    lightBoxImage.onclick = closeModal;
    lightBoxImage.tabIndex = 0;
    lightBoxModal.appendChild(lightBoxImage);

    const modalPrev = document.createElement('a');
    modalPrev.className = 'modalPrev';
    modalPrev.innerHTML = '&#10094;';
    modalPrev.tabIndex = 0;
    modalPrev.addEventListener('click', modalPrevImage, true);
    lightBoxModal.appendChild(modalPrev);

    const modalNext = document.createElement('a');
    modalNext.className = 'modalNext';
    modalNext.innerHTML = '&#10095;';
    modalNext.tabIndex = 0;
    modalNext.addEventListener('click', modalNextImage, true);
    lightBoxModal.appendChild(modalNext);

    // document.body.appendChild(modal);
    gradioApp().appendChild(lightBoxModal);

});
