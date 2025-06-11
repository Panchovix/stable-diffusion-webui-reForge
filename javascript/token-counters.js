let promptTokenCountUpdateFunctions = {};


function setupTokenCounting(id, id_counter, id_button) {
    var prompt = gradioApp().getElementById(id);
    var counter = gradioApp().getElementById(id_counter);
    var button = gradioApp().getElementById(id_button);
    var textarea = gradioApp().querySelector(`#${id} > label > textarea`);

    if (counter.parentElement == prompt.parentElement) {
        return;
    }

    prompt.parentElement.insertBefore(counter, prompt);
    prompt.parentElement.style.position = "relative";

    var func = onEdit(id, textarea, 800, function() {
        button?.click();
    });
    promptTokenCountUpdateFunctions[id] = func;
    promptTokenCountUpdateFunctions[id_button] = func;
}

function runCodeForTokenCounters(fun) {
    fun('txt2img_prompt', 'txt2img_token_counter', 'txt2img_token_button');
    fun('txt2img_neg_prompt', 'txt2img_negative_token_counter', 'txt2img_negative_token_button');
    fun('img2img_prompt', 'img2img_token_counter', 'img2img_token_button');
    fun('img2img_neg_prompt', 'img2img_negative_token_counter', 'img2img_negative_token_button');
}

onUiLoaded(function() {
    if (!opts.disable_token_counters) {
		runCodeForTokenCounters(setupTokenCounting);
	}
});
