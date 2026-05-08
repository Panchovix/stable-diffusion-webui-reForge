function onCalcResolutionHires(enable, width, height, hr_scale, hr_resize_x, hr_resize_y) {
    function setInactive(elem, inactive) {
        elem.classList.toggle('inactive', !!inactive);
    }

    var hrUpscaleBy = gradioApp().getElementById('txt2img_hr_scale');
    var hrResizeX = gradioApp().getElementById('txt2img_hr_resize_x');
    var hrResizeY = gradioApp().getElementById('txt2img_hr_resize_y');
    var hrCfgScale = gradioApp().getElementById('txt2img_hr_cfg');

    setInactive(hrUpscaleBy, opts.use_old_hires_fix_width_height || hr_resize_x > 0 || hr_resize_y > 0);
    setInactive(hrResizeX, opts.use_old_hires_fix_width_height || hr_resize_x == 0);
    setInactive(hrResizeY, opts.use_old_hires_fix_width_height || hr_resize_y == 0);

    return [enable, width, height, hr_scale, hr_resize_x, hr_resize_y];
}

function updateHrCfgScaleState() {
    const hrCfgScale = gradioApp().getElementById('txt2img_hr_cfg');
    if (hrCfgScale) {
        const input = hrCfgScale.querySelector('input');
        if (input) {
            const value = parseFloat(input.value);
            hrCfgScale.classList.toggle('inactive', value === 0);
        }
    }
}

document.addEventListener('DOMContentLoaded', function() {
    const hrCfgScale = gradioApp().getElementById('txt2img_hr_cfg');
    if (hrCfgScale) {
        const input = hrCfgScale.querySelector('input');
        if (input) {
            input.addEventListener('input', updateHrCfgScaleState);
            input.addEventListener('change', updateHrCfgScaleState);
            
            updateHrCfgScaleState();
        }
    }
});

onUiUpdate(function() {
    updateHrCfgScaleState();
});