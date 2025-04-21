//	combined edit attention and edit order

function keyupEditAttentionandOrder(event) {
    let target = event.originalTarget || event.composedPath()[0];

    if (event.metaKey || event.ctrlKey)
	{	//	ctrl + up/down arrows change attention
		let isPlus = event.key == "ArrowUp";
		let isMinus = event.key == "ArrowDown";
		if (!isPlus && !isMinus) return;

		let selectionStart = target.selectionStart;
		let selectionEnd = target.selectionEnd;
		let text = target.value;

		function selectCurrentParenthesisBlock(OPEN, CLOSE) {
			// Find opening parenthesis around current cursor
			const before = text.substring(0, selectionStart);
			let beforeParen = before.lastIndexOf(OPEN);
			if (beforeParen == -1) return false;

			let beforeClosingParen = before.lastIndexOf(CLOSE);
			if (beforeClosingParen != -1 && beforeClosingParen > beforeParen) return false;

			// Find closing parenthesis around current cursor
			const after = text.substring(selectionStart);
			let afterParen = after.indexOf(CLOSE);
			if (afterParen == -1) return false;

			let afterOpeningParen = after.indexOf(OPEN);
			if (afterOpeningParen != -1 && afterOpeningParen < afterParen) return false;

			// Set the selection to the text between the parenthesis
			const parenContent = text.substring(beforeParen + 1, selectionStart + afterParen);
			if (/.*:-?[\d.]+/s.test(parenContent)) {
				const lastColon = parenContent.lastIndexOf(":");
				selectionStart = beforeParen + 1;
				selectionEnd = selectionStart + lastColon;
			} else {
				selectionStart = beforeParen + 1;
				selectionEnd = selectionStart + parenContent.length;
			}

			target.setSelectionRange(selectionStart, selectionEnd);
			return true;
		}

		if (selectionStart == selectionEnd) {
		// If the user hasn't selected anything, let's select their current parenthesis block or word
			if (!selectCurrentParenthesisBlock('<', '>') && !selectCurrentParenthesisBlock('(', ')') && !selectCurrentParenthesisBlock('[', ']')) {
				const selection = window.getSelection();
				// Move cursor to beginning of the word
				selection?.modify('move', 'backward', 'word');
				// Extend selection to the end of the word
				selection?.modify('extend', 'forward', 'word');
				selectionStart = target.selectionStart;
				selectionEnd = target.selectionEnd;
			}
		}

		event.preventDefault();

		var closeCharacter = ')';
		var delta = opts.keyedit_precision_attention;
		var start = selectionStart > 0 ? text[selectionStart - 1] : "";
		var end = text[selectionEnd];

		if (start == '<') {
			closeCharacter = '>';
			delta = opts.keyedit_precision_extra;
		} else if (start == '(' && end == ')' || start == '[' && end == ']') { // convert old-style (((emphasis)))
			let numParen = 0;

			while (text[selectionStart - numParen - 1] == start && text[selectionEnd + numParen] == end) {
				numParen++;
			}

			if (start == "[") {
				weight = (1 / 1.1) ** numParen;
			} else {
				weight = 1.1 ** numParen;
			}

			weight = Math.round(weight / opts.keyedit_precision_attention) * opts.keyedit_precision_attention;

			text = text.slice(0, selectionStart - numParen) + "(" + text.slice(selectionStart, selectionEnd) + ":" + weight + ")" + text.slice(selectionEnd + numParen);
			selectionStart -= numParen - 1;
			selectionEnd -= numParen - 1;
		} else if (start != '(') {
			// do not include spaces at the end
			while (selectionEnd > selectionStart && text[selectionEnd - 1] == ' ') {
				selectionEnd--;
			}

			if (selectionStart == selectionEnd) {
				return;
			}

			text = text.slice(0, selectionStart) + "(" + text.slice(selectionStart, selectionEnd) + ":1.0)" + text.slice(selectionEnd);

			selectionStart++;
			selectionEnd++;
		}

		if (text[selectionEnd] != ':') return;
		var weightLength = text.slice(selectionEnd + 1).indexOf(closeCharacter) + 1;
		var weight = parseFloat(text.slice(selectionEnd + 1, selectionEnd + weightLength));
		if (isNaN(weight)) return;

		weight += isPlus ? delta : -delta;
		weight = parseFloat(weight.toPrecision(12));
		if (Number.isInteger(weight)) weight += ".0";

		if (closeCharacter == ')' && weight == 1) {
			var endParenPos = text.substring(selectionEnd).indexOf(')');
			text = text.slice(0, selectionStart - 1) + text.slice(selectionStart, selectionEnd) + text.slice(selectionEnd + endParenPos + 1);
			selectionStart--;
			selectionEnd--;
		} else {
			text = text.slice(0, selectionEnd + 1) + weight + text.slice(selectionEnd + weightLength);
		}

		target.focus();
		target.value = text;
		target.selectionStart = selectionStart;
		target.selectionEnd = selectionEnd;

		updateInput(target);
	}
	else if (event.altKey && opts.keyedit_move)
	{	// alt + left/right arrows moves text in prompt
		let isLeft = event.key == "ArrowLeft";
		let isRight = event.key == "ArrowRight";
		if (!isLeft && !isRight) return;
		event.preventDefault();

		let selectionStart = target.selectionStart;
		let selectionEnd = target.selectionEnd;
		let text = target.value;
		let items = text.split(",");
		let indexStart = (text.slice(0, selectionStart).match(/,/g) || []).length;
		let indexEnd = (text.slice(0, selectionEnd).match(/,/g) || []).length;
		let range = indexEnd - indexStart + 1;

		if (isLeft && indexStart > 0) {
			items.splice(indexStart - 1, 0, ...items.splice(indexStart, range));
			target.value = items.join();
			target.selectionStart = items.slice(0, indexStart - 1).join().length + (indexStart == 1 ? 0 : 1);
			target.selectionEnd = items.slice(0, indexEnd).join().length;
		}
		else if (isRight && indexEnd < items.length - 1) {
			items.splice(indexStart + 1, 0, ...items.splice(indexStart, range));
			target.value = items.join();
			target.selectionStart = items.slice(0, indexStart + 1).join().length + 1;
			target.selectionEnd = items.slice(0, indexEnd + 2).join().length;
		}

		event.preventDefault();
		updateInput(target);
	}
}

onUiLoaded(function() {
	const prompt_boxes = ['txt2img_prompt', 'txt2img_neg_prompt', 'img2img_prompt', 'img2img_neg_prompt'];
	prompt_boxes.forEach(function(prompt) {
		var textarea = gradioApp().querySelector("#" + prompt + " > label > textarea");

		textarea.addEventListener('keydown', (event) => {
			keyupEditAttentionandOrder(event);
		});
	});
});
